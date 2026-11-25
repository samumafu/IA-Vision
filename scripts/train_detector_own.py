"""Entrena detectores VisionLocaliza y comprueba una iteración con --smoke."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.detector_own import DETECTORS, detection_loss, decode_predictions
from training.detection_data import DetectionDataset, collate_detection
from training.detection_metrics import nms, spatial_metrics, coco_reference


def atomic_save(payload,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audited_training_extension(root,previous_sha):
    """Allow only an append of verified human-annotated training images.

    The exact original rows, validation and reserved test must be preserved.
    A changed audit alone is never enough to continue a checkpoint.
    """
    root=Path(root)
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    parent_bytes=(root/'parent_manifest.json').read_bytes()
    if (audit.get('parent_audit_sha256')!=previous_sha or
        hashlib.sha256((root/'parent_audit.json').read_bytes()).hexdigest()!=previous_sha or
        hashlib.sha256(parent_bytes).hexdigest()!=audit.get('parent_manifest_sha256')):
        raise ValueError('La ampliación no corresponde a los datos originales.')
    parent=json.loads(parent_bytes)
    current=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    extra=current[len(parent):]
    if (current[:len(parent)]!=parent or not extra or
        any(r['split']!='train' or r.get('source_original_split')!='train' or
            r.get('source_dataset')!='MJU-Waste' or r.get('human_polygon_mask_IoU',0)<.95 for r in extra) or
        len({r['id'] for r in current})!=len(current) or
        audit.get('cross_split_candidates')!=0 or audit.get('foundation_annotations_used') is not False):
        raise ValueError('Solo se admite ampliar train con anotaciones humanas auditadas.')
    if [r for r in current if r['split']!='train']!=[r for r in parent if r['split']!='train']:
        raise ValueError('Las particiones reservadas cambiaron.')
    return {'previous_audit_sha256':previous_sha,
            'dataset_audit_sha256':hashlib.sha256((root/'audit.json').read_bytes()).hexdigest(),
            'added_train_images':len(extra),'source':audit['source'],
            'validation_and_test_unchanged':True}


def validate(model,loader,device,near_ids=()):
    model.eval();records=[]
    for images,boxes,ids,_,_ in loader:
        with torch.inference_mode():predictions=decode_predictions(model(images.to(device)),confidence=.001,strides=model.strides)
        size=images.shape[-1]
        for ident,actual,(pb,ps) in zip(ids,boxes,predictions):
            pb=pb.cpu().numpy().clip(0,size);ps=ps.cpu().numpy()
            keep=nms(pb,ps,.45,max_detections=100)
            truth=[{'box':b.tolist(),'category':'residuo'} for b in actual]
            records.append({'image_id':ident,'size':[size,size],'truth':truth,
                            'predictions':[{'box':pb[i].tolist(),'score':float(ps[i])} for i in keep]})
    result=spatial_metrics(records,confidence=.35)
    reference=coco_reference(records)
    if max(abs(result[k]-reference[k]) for k in reference)>1e-6:raise RuntimeError('AP difiere de COCO.')
    result['COCO_reference']=reference
    if near_ids:
        fixed=[r for r in records if r['image_id'] not in near_ids]
        near=[r for r in records if r['image_id'] in near_ids]
        result['fixed_validation']=spatial_metrics(fixed,confidence=.35)
        result['near_validation']=spatial_metrics(near,confidence=.35)
    return result


def train(config,root,output,smoke=False,resume=None):
    config=json.loads(Path(config).read_text(encoding='utf-8'))
    architecture = config['architecture']
    if architecture not in DETECTORS or config['pretrained'] is not False:
        raise ValueError('Se exige arquitectura propia y pesos aleatorios.')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    cv2.setNumThreads(0);torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not smoke and device.type!='cuda':raise RuntimeError('Entrenamiento largo: usar T4, no CPU.')
    if device.type=='cuda':
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.benchmark=True
    run=Path(output).resolve()
    if run.exists() and not resume:raise ValueError('La ejecución ya existe; no sobrescribirla.')
    run.mkdir(parents=True,exist_ok=True)
    model=DETECTORS[architecture]().to(device, memory_format=torch.channels_last)
    collate=partial(collate_detection, strides=model.strides)
    initial_hash=fingerprint(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    amp=config.get('native_amp',True) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=512.)
    ema=deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    provenance={'architecture':architecture,'initialization':'random','pretrained':False,
                'GPU':torch.cuda.get_device_name() if device.type=='cuda' else None,
                'official_detector_components':False,'initial_state_sha256':initial_hash,
                'design_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'training/detector_own.py').read_bytes()).hexdigest(),
                'parameters':sum(p.numel() for p in model.parameters()),
                'dataset_audit_sha256':hashlib.sha256((Path(root)/'audit.json').read_bytes()).hexdigest(),
                'test_used_for_selection':False,'config':config}
    ap50_weight=config.get('selection_ap50_weight',0.)
    selection='validation mAP50:95' if not ap50_weight else f'validation {ap50_weight:g} mAP50 + {1-ap50_weight:g} mAP50:95'
    start_epoch,best,best_ap95,best_epoch=0,-1.,-1.,0
    if resume:
        ck=torch.load(resume,map_location=device,weights_only=True)
        if ck['architecture']!=architecture or ck['provenance']['pretrained']:raise ValueError('Checkpoint ajeno/incompatible.')
        changed={k for k in set(ck['config'])|set(config) if ck['config'].get(k)!=config.get(k)}
        allowed={'object_crop_probability','background_crop_probability','focus_categories','native_amp'}
        extension=None
        if ck['provenance']['dataset_audit_sha256']!=provenance['dataset_audit_sha256']:
            extension=audited_training_extension(root,ck['provenance']['dataset_audit_sha256'])
            allowed.add('epochs')  # More iterations for the audited additional train images.
        if changed-allowed or ('epochs' in changed and config['epochs']<ck['config']['epochs']):
            raise ValueError('Cambio de arquitectura u optimización no autorizado por esta continuación.')
        model.load_state_dict(ck['state_dict']);ema.load_state_dict(ck['ema_state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        if ck['scaler'] and amp:scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch'];best_ap95=ck['best_mAP50_95'];best=ck.get('best_selection_score',best_ap95)
        best_epoch=ck['best_epoch'];provenance=ck['provenance']
        if extension:
            provenance.setdefault('data_phases',[]).append(dict(extension,starting_epoch=start_epoch+1,
                continued_own_checkpoint_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest()))
            provenance['dataset_audit_sha256']=extension['dataset_audit_sha256']
        if changed:
            provenance.setdefault('augmentation_phases',[]).append({'starting_epoch':start_epoch+1,
                'previous':{k:ck['config'].get(k) for k in sorted(changed)},
                'current':{k:config.get(k) for k in sorted(changed)},
                'continued_own_checkpoint_sha256':hashlib.sha256(Path(resume).read_bytes()).hexdigest()})
            provenance['config']=config
        torch.set_rng_state(ck['torch_rng'].cpu())
        random.setstate(ck['python_rng'])
        if device.type=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng']])
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    dataset=DetectionDataset(root,'train',config,limit=2 if smoke else None)
    if smoke:
        batch=collate([dataset[i] for i in range(2)])
        images,boxes,_,targets,collisions=batch
        prepared=[{k:v.to(device) for k,v in target.items()} for target in targets]
        model.train();before=fingerprint(model)
        with torch.autocast(device.type,enabled=amp):predictions=model(images.to(device))
        loss,parts=detection_loss(predictions,prepared_targets=(prepared,collisions),strides=model.strides)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not torch.isfinite(loss) or not finite:raise RuntimeError('Pérdida o gradientes no finitos.')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        scaler.step(optimizer);scaler.update()
        after=fingerprint(model)
        if before==after:raise RuntimeError('El optimizador no actualizó los pesos.')
        payload={'architecture':architecture,'state_dict':model.state_dict(),'config':config,'provenance':provenance,'image_size':config['image_size']}
        atomic_save(payload,run/'iteration.pth')
        restored=DETECTORS[architecture]().to(device, memory_format=torch.channels_last)
        restored.load_state_dict(torch.load(run/'iteration.pth',map_location=device,weights_only=True)['state_dict'])
        model.eval();restored.eval()
        with torch.inference_mode():
            expected=model(images.to(device));observed=restored(images.to(device))
        error=max(float((a-b).abs().max()) for a,b in zip(expected,observed))
        assert error==0., error
        result={'passed':True,'real_annotated_images':2,'device':str(device),'parameters':provenance['parameters'],
                'loss':float(loss.detach()),'parts':{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()},'finite_gradients':finite,
                'weights_changed':before!=after,'checkpoint_reload_max_error':error,
                'decoded_images':len(decode_predictions(observed,strides=model.strides)),'test_used':False,
                'note':'Verifies a complete training iteration, not detection quality.'}
        (run/'smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True);return result
    passed_smokes=[]
    for path in run.parent.glob('smoke*/smoke.json'):
        result=json.loads(path.read_text())
        details=json.loads((path.parent/'provenance.json').read_text())
        if (result['passed'] and result['device']==str(device) and details['config']==config and
            details['dataset_audit_sha256']==provenance['dataset_audit_sha256']):
            passed_smokes.append(path)
    if not passed_smokes:raise RuntimeError('Primero ejecutar --smoke con la misma configuración y GPU.')
    valset=DetectionDataset(root,'val',config)
    near_sources=set(config.get('near_validation_sources', ()))
    near_ids={row['id'] for row in valset.rows if row.get('source_dataset') in near_sources}
    near_weight=config.get('selection_near_weight', 0.)
    if near_weight and (not 0 < near_weight < 1 or not 0 < len(near_ids) < len(valset)):
        raise ValueError('Near-object selection requires distinct general and near validation scenes.')
    if near_weight:
        selection += f', {near_weight:g} weight on separate near validation'
    sampler = None
    fractions=config.get('training_source_fractions')
    if fractions:
        domains=[r.get('source_dataset', 'TACO') for r in dataset.rows]
        if set(fractions)!=set(domains) or not math.isclose(sum(fractions.values()), 1.) or min(fractions.values())<=0:
            raise ValueError('Source fractions must match the manifest and sum to 1.')
        focus=config.get('training_box_focus', {})
        if any(weight<1 for weight in focus.values()):
            raise ValueError('Object focus must preserve or increase the weight of matching scenes.')
        strengths=[max((focus.get(box['category'], 1.) for box in row['boxes']), default=1.) for row in dataset.rows]
        totals=Counter()
        for source,strength in zip(domains,strengths):
            totals[source]+=strength
        weights=[fractions[source]*strength/totals[source] for source,strength in zip(domains,strengths)]
        sampler=WeightedRandomSampler(weights,config['epoch_samples'],replacement=True)
    elif 'swim3_fraction' in config:
        swim3 = [row.get('source_dataset') == 'SWiM3' for row in dataset.rows]
        size = sum(swim3)
        if not 0 < size < len(dataset):
            raise ValueError('SWiM3 sampling requires both annotation sources.')
        fraction = config['swim3_fraction']
        weights = [fraction / size if flag else (1 - fraction) / (len(dataset) - size) for flag in swim3]
        sampler = WeightedRandomSampler(weights, config['epoch_samples'], replacement=True)
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=sampler is None,sampler=sampler,num_workers=config['workers'],
                      collate_fn=collate,pin_memory=True,persistent_workers=config['workers']>0)
    validation=DataLoader(valset,batch_size=config['batch_size'],shuffle=False,num_workers=0,collate_fn=collate)
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
        print('Detención solicitada: guardar al completar la época.',flush=True)
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    history=[]
    if (run/'history.json').exists():history=json.loads((run/'history.json').read_text())
    last_metrics=None
    for epoch in range(start_epoch,config['epochs']):
        started=time.monotonic();model.train();loss_sum=0.;collisions_sum=0
        if sampler is not None:
            sampler.generator = torch.Generator().manual_seed(config['seed'] + epoch)
        warmup=min(1.,(epoch+1)/config['warmup_epochs'])
        progress=max(0.,(epoch-config['warmup_epochs'])/max(1,config['epochs']-config['warmup_epochs']))
        lr=config['learning_rate']*warmup*(.05+.95*(1+math.cos(math.pi*progress))/2)
        for group in optimizer.param_groups:group['lr']=lr
        for images,boxes,ids,targets,collisions in loader:
            images=images.to(device,non_blocking=True,memory_format=torch.channels_last)
            prepared=[{k:v.to(device,non_blocking=True) for k,v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=amp):outputs=model(images)
            loss,parts=detection_loss(outputs,prepared_targets=(prepared,collisions),strides=model.strides)
            if not torch.isfinite(loss):raise RuntimeError(f'Pérdida no finita en época {epoch+1}. Se conserva last.pth.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(norm):raise RuntimeError('Gradientes no finitos; no sobrescribir last.pth.')
            scaler.step(optimizer);scaler.update()
            with torch.no_grad():
                decay=min(.999,1-1/(epoch*len(loader)+len(history)+10))
                for ema_value,value in zip(ema.state_dict().values(),model.state_dict().values()):
                    if ema_value.dtype.is_floating_point:ema_value.mul_(decay).add_(value.detach(),alpha=1-decay)
                    else:ema_value.copy_(value)
            loss_sum+=float(loss.detach());collisions_sum+=collisions
        is_validation=(epoch==0 or (epoch+1)%config['validation_every']==0 or epoch+1==config['epochs'] or stop_requested)
        if is_validation:
            last_metrics=validate(ema,validation,device,near_ids)
            general=last_metrics.get('fixed_validation',last_metrics)
            selection_score=ap50_weight*general['mAP50']+(1-ap50_weight)*general['mAP50_95']
            if near_weight:
                near=last_metrics['near_validation']
                near_score=ap50_weight*near['mAP50']+(1-ap50_weight)*near['mAP50_95']
                selection_score=(1-near_weight)*selection_score+near_weight*near_score
            if selection_score>best:
                best=selection_score;best_ap95=last_metrics['mAP50_95'];best_epoch=epoch+1
                atomic_save({'architecture':architecture,'state_dict':ema.state_dict(),'config':config,
                             'provenance':provenance,'image_size':config['image_size'],'epoch':epoch+1,
                             'validation':last_metrics},run/'best.pth')
        record={'epoch':epoch+1,'loss':loss_sum/len(loader),'lr':lr,'seconds':time.monotonic()-started,
                'center_collisions':collisions_sum,'validation':last_metrics if is_validation else None}
        history.append(record)
        state={'architecture':architecture,'state_dict':model.state_dict(),'ema_state_dict':ema.state_dict(),
               'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'config':config,
               'provenance':provenance,'best_mAP50_95':best_ap95,'best_selection_score':best,'best_epoch':best_epoch,
               'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        atomic_save(state,run/'last.pth')
        if (epoch+1)%20==0:atomic_save(state,run/f'epoch_{epoch+1:03d}.pth')
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        status={'run':run.name,'epoch':epoch+1,'best_epoch':best_epoch,'best_mAP50_95':best_ap95,'best_selection_score':best,
                'last_validation':last_metrics,'stopped':stop_requested,'complete':False}
        (run.parent/'status.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(record),flush=True)
        if stop_requested or epoch+1-best_epoch>=config['patience']:break
    status['complete']=not stop_requested
    (run.parent/'status.json').write_text(json.dumps(status,indent=2))
    (run.parent/'selected.json').write_text(json.dumps({'run':run.name,'checkpoint':'best.pth',
           'architecture':architecture,'best_epoch':best_epoch,'selection':selection,
           'test_used':False,'stopped':stop_requested},indent=2))
    print('FINISHED',run,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/detector_D2.json');p.add_argument('--root',default='data/detection/taco')
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume')
    a=p.parse_args();train(a.config,a.root,a.output,a.smoke,a.resume)
