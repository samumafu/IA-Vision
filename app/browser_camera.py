"""Browser camera frames sent through Streamlit's existing HTTPS/WebSocket channel."""
import base64
from collections import deque
from pathlib import Path
import time

import streamlit.components.v1 as components

from app.detector import LIVE_TRACK_MAX_AGE, ObjectTracker, box_iou, center_detection
from app.recycling import guidance
from app.vision import decode_image

camera_component=components.declare_component('vision_camera',path=str(Path(__file__).parent/'camera_component'))
CAMERA_GUIDE=(.16,.10,.84,.90)
CAMERA_CANDIDATE_CONFIDENCE=.20


class BrowserRecognition:
    """Session-local tracking. No camera hardware, disk writes or remote models."""
    def __init__(self, material_threshold=.65):
        self.tracker=ObjectTracker(live=True)
        self.material_threshold=material_threshold
        self.stream_id=None
        self.sequence=0
        self.retired=deque(maxlen=8)
        self.objects=()
        self.response={}
        self.running=False
        self.pending_candidate=None
        self.zoom_retry_at=0

    def _tracked_box(self, hint, frame):
        if not isinstance(hint,dict) or type(hint.get('track_id')) is not int:
            return None
        if hint['track_id'] not in self.tracker.tracks or time.monotonic()-self.tracker.last_seen>LIVE_TRACK_MAX_AGE:
            return None
        box=hint.get('box')
        if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not 0<=v<=1 for v in box):
            return None
        x1,y1,x2,y2=box
        if x2<=x1 or y2<=y1:return None
        height,width=frame.shape[:2]
        return (x1*width,y1*height,x2*width,y2*height)

    def _detect_live(self, frame, detector, tracked_box=None):
        threshold=getattr(detector,'confidence',.25)
        minimum=min(threshold,CAMERA_CANDIDATE_CONFIDENCE)
        detect=getattr(detector,'detect_close',detector.detect)
        proposals=detect(frame,confidence=minimum)
        detections=[item for item in proposals if item.confidence>=threshold]
        if detections:
            self.pending_candidate=None
            self.zoom_retry_at=0
            return detections

        now=time.monotonic()
        height,width=frame.shape[:2]
        active=next(iter(self.tracker.tracks.values()),None)
        reference=tracked_box if tracked_box is not None else active['box'] if active and now-self.tracker.last_seen<1.1 else None
        if reference is not None:
            for proposal in proposals:
                if box_iou(proposal.box,reference)>=.5:
                    self.pending_candidate=None
                    return [proposal]

        gx1,gy1,gx2,gy2=CAMERA_GUIDE
        def inside_guide(item):
            x1,y1,x2,y2=item.box
            return gx1*width<=(x1+x2)/2<=gx2*width and gy1*height<=(y1+y2)/2<=gy2*height

        candidate=next((item for item in proposals if inside_guide(item)),None)
        if self.sequence>=self.zoom_retry_at:
            zoom=self._center_candidate(frame,detector,minimum)
            if zoom and (zoom.confidence>=threshold or candidate is None):
                candidate=zoom
            if zoom is None:
                self.zoom_retry_at=self.sequence+3

        if candidate is None:
            self.pending_candidate=None
            return []
        previous=self.pending_candidate
        readings=1
        if previous and self.sequence==previous[2]+1 and now-previous[1]<1.1:
            if box_iou(candidate.box,previous[0])>=.5:
                readings=previous[3]+1
        self.pending_candidate=(candidate.box,now,self.sequence,readings)
        if reference is not None and box_iou(candidate.box,reference)>=.5:
            return [candidate]
        required=2 if candidate.confidence>=threshold else 3
        return [candidate] if readings>=required else []

    @staticmethod
    def _center_candidate(frame, detector, minimum):
        return center_detection(frame,detector.detect,minimum)

    def consume(self,payload,detector,classifier):
        if not isinstance(payload,dict):return False
        sid,sequence=payload.get('stream_id'),payload.get('sequence')
        if not isinstance(sid,str) or not 1<=len(sid)<=80 or type(sequence) is not int or sequence<1:return False
        if sid in self.retired:return False
        if sid==self.stream_id and sequence<=self.sequence:return False
        if sid!=self.stream_id:
            if self.stream_id:self.retired.append(self.stream_id)
            self.stream_id=sid;self.sequence=0;self.tracker=ObjectTracker(live=True);self.objects=()
            self.pending_candidate=None;self.zoom_retry_at=0
        self.sequence=sequence
        self.response={'stream_id':sid,'sequence':sequence,'detection':None}
        if payload.get('event')=='stopped':
            self.running=False;self.objects=();self.tracker=ObjectTracker(live=True)
            self.pending_candidate=None;self.zoom_retry_at=0
            return True
        try:
            if payload.get('event')!='frame':raise ValueError('Fotograma inválido.')
            encoded=payload.get('image')
            if not isinstance(encoded,str) or len(encoded)>2_000_000:raise ValueError('Fotograma demasiado grande.')
            frame=decode_image(base64.b64decode(encoded,validate=True),max_bytes=1_500_000,max_pixels=640*640)
            tracked_box=self._tracked_box(payload.get('tracking'),frame)
            objects=self.tracker.update(frame,self._detect_live(frame,detector,tracked_box),classifier,tracked_box=tracked_box)
            self.objects=tuple(objects);self.running=True
            if objects:
                item=objects[0];height,width=frame.shape[:2]
                box=[float(v)/scale for v,scale in zip(item.detection.box,[width,height,width,height])]
                result=None
                if item.confirmed:
                    reading=item.reading
                    info=guidance(reading.material)
                    label=info['label']
                    if reading.confidence<self.material_threshold:label='Posible '+label.lower()
                    result={'label':label,'destination':info['destination'],
                            'organic':reading.material=='organicos'}
                self.response['detection']={'box':box,'track_id':item.track_id,'material':result}
        except Exception:
            self.objects=();self.tracker=ObjectTracker(live=True)
            self.pending_candidate=None;self.zoom_retry_at=0
            self.response['error']='No se pudo reconocer este fotograma. Prueba con más luz y el residuo completo.'
        return True
