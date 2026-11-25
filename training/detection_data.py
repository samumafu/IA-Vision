"""Anotaciones humanas TACO y aumentaciones geométricas con cajas transformadas."""
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from training.detector_own import STRIDES, make_targets


def letterbox(rgb,size,boxes=None,training=False,config=None):
    config=config or {}
    height,width=rgb.shape[:2]
    scale=min(size/width,size/height)
    if training:scale*=random.uniform(*config.get('scale_range',[.65,1.25]))
    nw,nh=max(1,round(width*scale)),max(1,round(height*scale))
    if training:
        left=random.randint(min(0,size-nw),max(0,size-nw))
        top=random.randint(min(0,size-nh),max(0,size-nh))
    else:left,top=(size-nw)//2,(size-nh)//2
    scaled=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.full((size,size,3),114,dtype=np.uint8)
    x1,y1=max(0,left),max(0,top);x2,y2=min(size,left+nw),min(size,top+nh)
    canvas[y1:y2,x1:x2]=scaled[y1-top:y2-top,x1-left:x2-left]
    transformed=None
    if boxes is not None:
        transformed=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy()
        transformed[:,[0,2]]=transformed[:,[0,2]]*(nw/width)+left
        transformed[:,[1,3]]=transformed[:,[1,3]]*(nh/height)+top
        transformed=np.clip(transformed,0,size)
        keep=((transformed[:,2]-transformed[:,0])>=2)&((transformed[:,3]-transformed[:,1])>=2)
        if training:transformed=transformed[keep]
        if training and random.random()<config.get('horizontal_flip',.5):
            canvas=canvas[:,::-1].copy()
            transformed[:,[0,2]]=size-transformed[:,[2,0]]
        if training:
            image=Image.fromarray(canvas)
            for key,enhancer in [('brightness',ImageEnhance.Brightness),('contrast',ImageEnhance.Contrast)]:
                amount=config.get(key,.25)
                image=enhancer(image).enhance(random.uniform(1-amount,1+amount))
            canvas=np.asarray(image).copy()
    return canvas,transformed,(nw/width,nh/height,left,top)


def image_tensor(rgb):
    return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/127.5-1.


def suppress_interior_texture(rgb,boxes,categories,config):
    """Debilita estampados interiores; conserva los bordes y las cajas humanas."""
    probability=config.get('interior_texture_probability',0.)
    if probability<=0:return rgb
    selected=[b for b,c in zip(boxes,categories) if c in config.get('texture_categories',[]) and random.random()<probability]
    if not selected:return rgb
    output=rgb.copy();height,width=rgb.shape[:2]
    for box in selected:
        x1,y1,x2,y2=box;dx,dy=(x2-x1)*.2,(y2-y1)*.2
        left,top=max(0,int(x1+dx)),max(0,int(y1+dy))
        right,bottom=min(width,int(x2-dx)),min(height,int(y2-dy))
        if min(right-left,bottom-top)<8:continue
        patch=output[top:bottom,left:right].astype(np.float32)
        grey=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
        grey=cv2.GaussianBlur(grey,(0,0),max(1.,min(grey.shape)/10))[:,:,None]
        # Transición suave, exactamente cero en el borde del parche interior.
        yy=np.sin(np.linspace(0,np.pi,bottom-top))**2
        xx=np.sin(np.linspace(0,np.pi,right-left))**2
        alpha=(yy[:,None]*xx[None,:]*random.uniform(.7,1.))[:,:,None]
        output[top:bottom,left:right]=np.clip(patch*(1-alpha)+grey*alpha,0,255).astype(np.uint8)
    return output


def annotated_crop(rgb,boxes,categories,config):
    """Vistas cercanas o contexto sin cajas, derivados solo de anotaciones humanas.

    Se transforman TODAS las cajas visibles. El rectángulo de la fotografía
    recortada nunca se convierte automáticamente en una caja de objeto.
    """
    height,width=rgb.shape[:2]
    roll=random.random()
    background=config.get('background_crop_probability',0.)
    rectangle=None
    if roll<background:
        for _ in range(20):
            side=max(32,round(min(width,height)*random.uniform(.3,.6)))
            left=random.randint(0,max(0,width-side));top=random.randint(0,max(0,height-side))
            candidate=np.array([left,top,min(width,left+side),min(height,top+side)],dtype=np.float32)
            overlap=(np.minimum(boxes[:,2:],candidate[2:])-np.maximum(boxes[:,:2],candidate[:2])).clip(0).prod(1)
            if not np.any(overlap>0):rectangle=candidate;break
    elif roll<background+config.get('object_crop_probability',0.) and len(boxes):
        preferred=[i for i,c in enumerate(categories) if c in config.get('focus_categories',[])]
        index=random.choice(preferred or list(range(len(boxes))))
        x1,y1,x2,y2=boxes[index]
        margin=random.uniform(.25,1.1)
        dx,dy=(x2-x1)*margin,(y2-y1)*margin
        rectangle=np.array([max(0,x1-dx),max(0,y1-dy),min(width,x2+dx),min(height,y2+dy)])
    if rectangle is None:return rgb,boxes
    left,top,right,bottom=map(int,rectangle)
    if min(right-left,bottom-top)<16:return rgb,boxes
    cropped=rgb[top:bottom,left:right].copy()
    shifted=boxes-np.array([left,top,left,top],dtype=np.float32)
    shifted[:,[0,2]]=shifted[:,[0,2]].clip(0,right-left)
    shifted[:,[1,3]]=shifted[:,[1,3]].clip(0,bottom-top)
    keep=((shifted[:,2]-shifted[:,0])>=2)&((shifted[:,3]-shifted[:,1])>=2)
    return cropped,shifted[keep]


class DetectionDataset(Dataset):
    def __init__(self,root,split,config,verify=True,limit=None):
        if split not in ('train','val'):raise ValueError('El desarrollo solo abre train/val.')
        self.root=Path(root);self.config=config;self.split=split
        self.rows=[r for r in json.loads((self.root/'manifest.json').read_text(encoding='utf-8')) if r['split']==split]
        if limit is not None:
            # Smoke on one original and one supplemental annotated image when present.
            supplemental=[r for r in self.rows if r.get('source_dataset')=='MJU-Waste']
            self.rows=([self.rows[0],supplemental[0]] if limit==2 and supplemental else self.rows[:limit])
        self.cache=[]
        for i,row in enumerate(self.rows):
            path=self.root/'images'/split/f"{row['id']:05d}.jpg"
            data=path.read_bytes()
            if verify and hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError(f'Imagen modificada: {path}')
            with Image.open(path) as source:rgb=np.asarray(source.convert('RGB'))
            h,w=rgb.shape[:2]
            limit=config.get('train_decode_max_side',640) if split=='train' else config['image_size']
            if max(h,w)>limit:
                ratio=limit/max(h,w)
                rgb=cv2.resize(rgb,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_AREA)
            self.cache.append(rgb)
            if (i+1)%200==0:print(f'{split}: {i+1}/{len(self.rows)} imágenes verificadas',flush=True)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];rgb=self.cache[index]
        h,w=rgb.shape[:2]
        boxes=np.array([b['xyxy_normalized'] for b in row['boxes']],dtype=np.float32)*[w,h,w,h]
        if self.split=='train':
            categories=[b['category'] for b in row['boxes']]
            rgb=suppress_interior_texture(rgb,boxes,categories,self.config)
            rgb,boxes=annotated_crop(rgb,boxes,categories,self.config)
        image,boxes,_=letterbox(rgb,self.config['image_size'],boxes,self.split=='train',self.config)
        return image_tensor(image),torch.as_tensor(boxes,dtype=torch.float32),row['id']


def collate_detection(batch,strides=STRIDES):
    images=torch.stack([b[0] for b in batch])
    boxes=[b[1] for b in batch]
    size=images.shape[-1]
    shapes=[torch.empty((len(batch),6,size//s,size//s)) for s in strides]
    targets,collisions=make_targets(shapes,boxes,strides)
    return images,boxes,[b[2] for b in batch],targets,collisions
