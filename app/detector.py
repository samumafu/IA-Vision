"""Localizador VisionLocaliza, NMS y recortes para las CNN propias."""
from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from app.close_box import refine_close_box
from app.localization import Region, square_box
from training.detector_own import DETECTORS, decode_predictions
from training.detection_data import letterbox, image_tensor
from training.detection_metrics import box_iou, fuse_detection_views, nms

LIVE_TRACK_MAX_AGE = 2.5


@dataclass(frozen=True)
class Detection:
    box: tuple
    confidence: float
    region: Region
    inference_ms: float = 0.


def crop_box(box,width,height,margin=.12,*,square=True):
    x1,y1,x2,y2=box
    dx,dy=max((x2-x1)*margin,(16-(x2-x1))/2),max((y2-y1)*margin,(16-(y2-y1))/2)
    expanded=(max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy))),
              min(width,int(np.ceil(x2+dx))),min(height,int(np.ceil(y2+dy))))
    if expanded[2]-expanded[0]<16 or expanded[3]-expanded[1]<16:
        left=min(max(0,expanded[0]),width-16)
        top=min(max(0,expanded[1]),height-16)
        expanded=(left,top,min(width,max(left+16,expanded[2])),min(height,max(top+16,expanded[3])))
    return square_box(expanded,width,height) if square else expanded


def center_detection(rgb, detect, minimum):
    height, width = rgb.shape[:2]
    left, top = round(width * .14), round(height * .14)
    zoom = rgb[top:height - top, left:width - left].copy()
    zoom_height, zoom_width = zoom.shape[:2]
    if min(zoom_height, zoom_width) < 32:
        return None
    margin = max(2, min(width, height) * .012)
    for detection in detect(zoom, confidence=minimum):
        x1, y1, x2, y2 = detection.box
        if min(x1, y1, zoom_width - x2, zoom_height - y2) < margin:
            continue
        box = x1 + left, y1 + top, x2 + left, y2 + top
        region = replace(detection.region, box=crop_box(box, width, height))
        return replace(detection, box=box, region=region)
    return None


class NeuralDetector:
    def __init__(self,checkpoint,cpu_threads=2,confidence=.35,nms_iou=.45,views=()):
        path=Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError('Falta el detector propio entrenado. Puedes usar el modo manual.')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        from utils.dataset import sha256
        self.checkpoint_sha256=sha256(path)
        provenance=payload.get('provenance',{})
        if provenance.get('pretrained') is not False or provenance.get('initialization')!='random':
            raise ValueError('Detector sin procedencia de entrenamiento desde cero.')
        self.architecture=payload.get('architecture')
        if self.architecture not in DETECTORS or provenance.get('official_detector_components') is not False:
            raise ValueError('Se exige una arquitectura propia VisionLocaliza compatible.')
        torch.set_num_threads(cpu_threads)
        self.cpu_threads=cpu_threads
        self.model=DETECTORS[self.architecture]()
        self.model.load_state_dict(payload['state_dict'],strict=True)
        self.model.eval().float()
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        self.size=int(payload['image_size'])
        self.confidence,self.nms_iou=confidence,nms_iou
        self.views=tuple((size, flip) for size, flip in views)
        if len(self.views) == 1 or any(type(size) is not int or size % 32 or not 128 <= size <= 1024
                                       or type(flip) is not bool for size, flip in self.views):
            raise ValueError('Las vistas deben tener tamaño múltiplo de 32 y un indicador de espejo.')
        self.provenance=provenance
        self._lock=threading.Lock()

    def detect(self,rgb,confidence=None):
        # Streamlit may invoke a cached model from a new script/callback thread.
        threads=getattr(self,'cpu_threads',2)
        if torch.get_num_threads()!=threads:torch.set_num_threads(threads)
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('Se requiere imagen RGB uint8.')
        started=time.perf_counter()
        height,width=rgb.shape[:2]
        threshold=self.confidence if confidence is None else confidence
        views=getattr(self,'views',())
        if views:
            predictions=fuse_detection_views([self._detect_view(rgb,size,flip,.001) for size,flip in views],
                                             final_nms=self.nms_iou,min_confidence=threshold)
        else:
            predictions=self._detect_view(rgb,self.size,False,threshold)
        elapsed=(time.perf_counter()-started)*1000
        return [Detection(tuple(p['box']),p['score'],
                          Region(crop_box(p['box'],width,height),f'neuronal-{self.architecture}'),elapsed)
                for p in predictions]

    def detect_close(self, rgb, confidence=None):
        """Expand the primary proposal when a nearby contour agrees with it."""
        started = time.perf_counter()
        detections = self.detect(rgb, confidence)
        if not detections or detections[0].confidence < self.confidence:
            return detections
        primary = detections[0]
        height, width = rgb.shape[:2]
        left, top, right, bottom = primary.box
        area = (right - left) * (bottom - top)
        edge_distance = min(left / width, top / height, 1 - right / width, 1 - bottom / height)
        if area >= .30 * height * width and edge_distance < .015:
            centered = center_detection(rgb, self.detect, max(self.confidence, primary.confidence - .065))
            if centered:
                x1, y1, x2, y2 = centered.box
                centered_area = (x2 - x1) * (y2 - y1)
                overlap = max(0, min(right, x2) - max(left, x1)) * max(0, min(bottom, y2) - max(top, y1))
                if centered_area <= .65 * area and overlap >= .75 * centered_area:
                    detections[0] = replace(centered, confidence=primary.confidence,
                                            inference_ms=(time.perf_counter() - started) * 1000)
                    return detections
        expanded = refine_close_box(rgb, primary.box)
        if expanded != primary.box:
            region = replace(primary.region, box=crop_box(expanded, width, height))
            detections[0] = replace(primary, box=expanded, region=region,
                                    inference_ms=(time.perf_counter() - started) * 1000)
        return detections

    def _detect_view(self,rgb,size,flip,confidence):
        height,width=rgb.shape[:2]
        canvas,_,(scale_x,scale_y,left,top)=letterbox(rgb,size)
        if flip:
            # Mirror the resized image, keeping the letterbox offsets and avoiding a full-frame copy.
            width_scaled=round(width*scale_x)
            picture=canvas[:,left:left+width_scaled]
            picture[:]=picture[:,::-1].copy()
        tensor=image_tensor(canvas).unsqueeze(0)
        with self._lock,torch.inference_mode():
            raw=self.model(tensor)
            decoded=decode_predictions(raw,confidence,strides=self.model.strides)[0]
            boxes,scores=decoded[0].cpu().numpy(),decoded[1].cpu().numpy()
        if len(scores)==0:
            return []
        boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/scale_x
        boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/scale_y
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
        valid=np.isfinite(boxes).all(axis=1)&np.isfinite(scores)&((boxes[:,2]-boxes[:,0])>=2)&((boxes[:,3]-boxes[:,1])>=2)
        boxes,scores=boxes[valid],scores[valid]
        keep=nms(boxes,scores,self.nms_iou,max_detections=100)
        predictions=[]
        for i in keep:
            x1,y1,x2,y2=map(float,boxes[i])
            box=(width-x2,y1,width-x1,y2) if flip else (x1,y1,x2,y2)
            predictions.append({'box':box,'score':float(scores[i])})
        return predictions


@dataclass(frozen=True)
class ObjectReading:
    detection: Detection
    reading: object
    track_id: int = 0
    confirmed: bool = True


def primary_detection(detections,previous_box=None):
    """One object; retain a nearby candidate when its score is close to the best."""
    if not detections:return None
    best=max(detections,key=lambda d:d.confidence)
    if previous_box is not None:
        nearby=[d for d in detections if box_iou(d.box,previous_box)>=.25]
        if nearby:
            current=max(nearby,key=lambda d:box_iou(d.box,previous_box))
            if current.confidence>=best.confidence-.10:return current
    return best


class ObjectTracker:
    """Un residuo principal estable; elimina pérdidas sin cajas fantasma."""
    def __init__(self, *, live=False):
        self.live=live
        self.tracks={}
        self.next_id=1
        self.misses=0
        self.last_seen=0.

    def update(self,frame,detections,classifier,*,tracked_box=None):
        from app.stability import LiveMaterialFilter, PredictionFilter
        from dataclasses import replace
        result,new={},{}
        used=set()
        now=time.monotonic()
        max_age=LIVE_TRACK_MAX_AGE if tracked_box is not None else 1.1
        if self.live and now-self.last_seen>max_age:self.tracks.clear()
        previous=next(iter(self.tracks.values()),None)
        previous_box=tracked_box if previous and tracked_box is not None else previous['box'] if previous else None
        selected=primary_detection(detections,previous_box)
        if self.live and not selected:
            self.misses+=1
            if self.misses>=2 and tracked_box is None:self.tracks.clear()
            return []
        self.misses=0
        self.last_seen=now
        for detection in ([selected] if selected is not None else []):
            candidates=[(box_iou(detection.box,previous_box),key) for key in self.tracks if key not in used]
            match=max(candidates,default=(0,None))
            key=match[1] if match[0]>=.25 else self.next_id
            if match[0]<.25:
                self.next_id+=1
            used.add(key)
            track=self.tracks.get(key)
            if track is None:
                track={'box':detection.box,'filter':LiveMaterialFilter() if self.live else PredictionFilter()}
            box=detection.box if self.live else tuple(.25*a+.75*b for a,b in zip(track['box'],detection.box))
            detection=replace(detection,box=box)
            reading=classifier.classify(frame,region=detection.region)
            temporal=track['filter'].update(frame,reading)
            result[key]=ObjectReading(detection,temporal or reading,key,temporal is not None)
            new[key]={'box':box,'filter':track['filter']}
        self.tracks=new
        return list(result.values())


def analyze_objects(frame,detector,classifier):
    selected=primary_detection(detector.detect(frame))
    return [ObjectReading(selected,classifier.classify(frame,region=selected.region))] if selected else []


def annotate_objects(frame,objects):
    from app.recycling import LABELS
    import unicodedata
    output=frame.copy()
    for item in objects:
        x1,y1,x2,y2=map(round,item.detection.box)
        color=(21,126,117) if item.confirmed else (215,140,20)
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        text=LABELS.get(item.reading.material,item.reading.material)
        text=unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode()
        cv2.putText(output,text,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
    return output
