"""VisionLocaliza-D2: diseño propio con capas básicas de PyTorch.

Cinco etapas convolucionales, contexto global y tres cabezas independientes.
No importa backbones, necks, cabezas ni pesos de bibliotecas de detección.
"""
import math
from bisect import bisect_right

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'VisionLocaliza-D2'
STRIDES = (8, 16, 32)
PYRAMID_ARCHITECTURE = 'VisionLocaliza-D3'
PYRAMID_STRIDES = (4, 8, 16, 32)


def conv(incoming, outgoing, kernel=3, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(incoming,outgoing,kernel,stride,
                                  dilation*(kernel//2),dilation=dilation,bias=False),
                         nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True))


class LocalizationHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local=conv(channels,48,kernel=1)
        self.mix=nn.Sequential(conv(64,48),conv(48,32,dilation=2))
        self.presence=nn.Conv2d(32,1,1)
        self.box=nn.Sequential(conv(32,24),nn.Conv2d(24,4,1))
        self.quality=nn.Conv2d(32,1,1)

    def forward(self,features,context):
        local=self.local(features)
        context=context.expand(-1,-1,*local.shape[-2:])
        mixed=self.mix(torch.cat((local,context),dim=1))
        return torch.cat((self.presence(mixed),self.box(mixed),self.quality(mixed)),dim=1)


class VisionLocaliza(nn.Module):
    strides = STRIDES

    def __init__(self):
        super().__init__()
        self.stem=conv(3,16,kernel=5,stride=2)
        self.detail=nn.Sequential(conv(16,32,stride=2),conv(32,32))
        self.small=nn.Sequential(conv(32,64,stride=2),conv(64,64))
        self.medium=nn.Sequential(conv(64,96,stride=2),conv(96,96))
        self.large=nn.Sequential(conv(96,128,stride=2),conv(128,128))
        self.context=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(128,16,1),nn.ReLU(inplace=True))
        self.heads=nn.ModuleList(LocalizationHead(c) for c in (64,96,128))
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.ones_(layer.weight);nn.init.zeros_(layer.bias)
        for head in self.heads:
            nn.init.constant_(head.presence.bias,math.log(.01/.99))
            nn.init.normal_(head.presence.weight,std=.01)
            nn.init.normal_(head.box[-1].weight,std=.01)
            nn.init.normal_(head.quality.weight,std=.01)

    def forward(self,images):
        detail=self.detail(self.stem(images))
        small=self.small(detail)
        medium=self.medium(small)
        large=self.large(medium)
        context=self.context(large)
        return [head(features,context) for head,features in zip(self.heads,(small,medium,large))]


class DetailHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                 nn.BatchNorm2d(channels), nn.ReLU(inplace=True), conv(channels, channels, 1))
        self.output = nn.Conv2d(channels, 6, 1)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[0] = math.log(.01 / .99)

    def forward(self, features):
        return self.output(self.mix(features))


class VisionLocalizaPyramid(nn.Module):
    """Fuse coarse context into a learned heatmap at strides 4, 8, 16 and 32."""

    strides = PYRAMID_STRIDES

    def __init__(self):
        super().__init__()
        self.stem = conv(3, 16, 5, stride=2)
        self.detail = nn.Sequential(conv(16, 24, stride=2), conv(24, 24))
        self.small = nn.Sequential(conv(24, 48, stride=2), conv(48, 48))
        self.medium = nn.Sequential(conv(48, 80, stride=2), conv(80, 80))
        self.large = nn.Sequential(conv(80, 112, stride=2), conv(112, 112, dilation=2))
        self.lateral = nn.ModuleList(conv(incoming, 24, 1) for incoming in (24, 48, 80, 112))
        self.heads = nn.ModuleList(DetailHead(24) for _ in self.strides)

    def forward(self, images):
        detail = self.detail(self.stem(images))
        small = self.small(detail)
        medium = self.medium(small)
        large = self.large(medium)
        features = [layer(value) for layer, value in zip(self.lateral, (detail, small, medium, large))]
        for index in (2, 1, 0):
            features[index] = features[index] + F.interpolate(features[index + 1], size=features[index].shape[-2:], mode='nearest')
        return [head(value) for head, value in zip(self.heads, features)]


DETECTORS = {ARCHITECTURE: VisionLocaliza, PYRAMID_ARCHITECTURE: VisionLocalizaPyramid}


def decode_head(raw,stride):
    """Cada posición predice centro dentro de su celda y anchura/altura positivas."""
    _,_,height,width=raw.shape
    y,x=torch.meshgrid(torch.arange(height,device=raw.device),torch.arange(width,device=raw.device),indexing='ij')
    grid=torch.stack((x,y)).to(raw.dtype).unsqueeze(0)
    center=(raw[:,1:3].sigmoid()+grid)*stride
    size=F.softplus(raw[:,3:5])*stride*4
    return torch.cat((center-size/2,center+size/2),dim=1)


def aligned_iou(first,second,generalized=False):
    intersection=(torch.minimum(first[:,2:],second[:,2:])-torch.maximum(first[:,:2],second[:,:2])).clamp(min=0).prod(1)
    area_first=(first[:,2:]-first[:,:2]).clamp(min=0).prod(1)
    area_second=(second[:,2:]-second[:,:2]).clamp(min=0).prod(1)
    union=(area_first+area_second-intersection).clamp(min=1e-6)
    iou=intersection/union
    if not generalized:return iou
    enclosing=(torch.maximum(first[:,2:],second[:,2:])-torch.minimum(first[:,:2],second[:,:2])).clamp(min=0).prod(1).clamp(min=1e-6)
    return iou-(enclosing-union)/enclosing


def make_targets(outputs,boxes,strides=STRIDES):
    """Asigna por área visible: pequeño <64 px, mediano <160 px, grande.

    Las cajas son anotaciones humanas transformadas; nunca se fabrican cajas.
    Si dos centros ocupan una celda, se conserva la caja menor y se registra.
    """
    targets=[]
    collisions=0
    for raw in outputs:
        b,_,h,w=raw.shape
        targets.append({'heat':raw.new_zeros((b,h,w)), 'boxes':raw.new_zeros((b,h,w,4)),
                        'positive':torch.zeros((b,h,w),dtype=torch.bool,device=raw.device),
                        'offset':raw.new_zeros((b,h,w,2))})
    for batch_index,image_boxes in enumerate(boxes):
        for box in image_boxes:
            width,height=(box[2:]-box[:2]).tolist()
            size=math.sqrt(max(width*height,1e-6))
            level=bisect_right((64, 160) if strides == STRIDES else (28, 70, 160), size)
            stride=strides[level]
            target=targets[level]
            center=(box[:2]+box[2:])/2/stride
            x=min(max(0,int(center[0])),target['heat'].shape[2]-1)
            y=min(max(0,int(center[1])),target['heat'].shape[1]-1)
            if target['positive'][batch_index,y,x]:
                collisions+=1
                old=target['boxes'][batch_index,y,x]
                if (old[2]-old[0])*(old[3]-old[1]) <= width*height:continue
            sigma=max(.5,min(2.,min(width,height)/stride/5))
            radius=math.ceil(3*sigma)
            x1,x2=max(0,x-radius),min(target['heat'].shape[2],x+radius+1)
            y1,y2=max(0,y-radius),min(target['heat'].shape[1],y+radius+1)
            yy,xx=torch.meshgrid(torch.arange(y1,y2,device=box.device),torch.arange(x1,x2,device=box.device),indexing='ij')
            gaussian=torch.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
            patch=target['heat'][batch_index,y1:y2,x1:x2]
            target['heat'][batch_index,y1:y2,x1:x2]=torch.maximum(patch,gaussian)
            target['positive'][batch_index,y,x]=True
            target['boxes'][batch_index,y,x]=box
            target['offset'][batch_index,y,x]=center-center.floor()
    return targets,collisions


def detection_loss(outputs,boxes=None,prepared_targets=None,strides=STRIDES):
    # Calcular las pérdidas en float32, incluso al usar autocast en la GPU.
    outputs=[o.float() for o in outputs]
    targets,collisions=prepared_targets if prepared_targets is not None else make_targets(outputs,boxes,strides)
    zero=sum(o.sum()*0 for o in outputs)
    heat,geometry,offset,size,quality=zero,zero,zero,zero,zero
    count=sum(int(t['positive'].sum()) for t in targets)
    for raw,target,stride in zip(outputs,targets,strides):
        positive=target['positive'];truth=target['heat']
        probability=raw[:,0].sigmoid().clamp(1e-5,1-1e-5)
        heat=heat+(-(1-probability).pow(2)*probability.log()*positive
                   -probability.pow(2)*(1-probability).log()*(1-truth).pow(4)*(~positive)).sum()
        if not positive.any():continue
        decoded=decode_head(raw,stride).permute(0,2,3,1)[positive]
        actual=target['boxes'][positive]
        geometry=geometry+(1-aligned_iou(decoded,actual,generalized=True)).sum()
        predicted_offset=raw[:,1:3].sigmoid().permute(0,2,3,1)[positive]
        offset=offset+F.smooth_l1_loss(predicted_offset,target['offset'][positive],reduction='sum')
        predicted_size=F.softplus(raw[:,3:5]).permute(0,2,3,1)[positive].clamp(min=1e-4).log()
        actual_size=((actual[:,2:]-actual[:,:2])/(stride*4)).clamp(min=1e-4).log()
        size=size+F.smooth_l1_loss(predicted_size,actual_size,reduction='sum')
        quality=quality+F.binary_cross_entropy_with_logits(raw[:,5][positive],aligned_iou(decoded.detach(),actual).detach(),reduction='sum')
    normalizer=max(count,1)
    losses={'presence':heat/normalizer,'GIoU':geometry/normalizer,'offset':offset/normalizer,
            'size':size/normalizer,'quality':quality/normalizer}
    total=losses['presence']+3*losses['GIoU']+losses['offset']+losses['size']+losses['quality']
    return total,dict(losses,positive_cells=count,collisions=collisions)


def decode_predictions(outputs,confidence=.05,max_candidates=1500,strides=STRIDES):
    """Devuelve cajas y puntuaciones por imagen; NMS se aplica fuera de la red."""
    results=[]
    decoded=[decode_head(o.float(),s).permute(0,2,3,1) for o,s in zip(outputs,strides)]
    scores_by_level = [o[:, 0].float().sigmoid() * o[:, 5].float().sigmoid() for o in outputs]
    if strides == PYRAMID_STRIDES:
        scores_by_level = [score * (score == F.max_pool2d(score[:, None], 3, 1, 1)[:, 0]) for score in scores_by_level]
    for index in range(outputs[0].shape[0]):
        all_boxes=[];all_scores=[]
        for level_scores,boxes in zip(scores_by_level,decoded):
            # Presencia de centro y calidad de localización son objetivos distintos.
            score=level_scores[index]
            mask=score>=confidence
            all_boxes.append(boxes[index][mask]);all_scores.append(score[mask])
        boxes=torch.cat(all_boxes);scores=torch.cat(all_scores)
        if len(scores)>max_candidates:
            scores,chosen=scores.topk(max_candidates);boxes=boxes[chosen]
        results.append((boxes,scores))
    return results
