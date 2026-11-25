"""Verifica geometría, evaluación y pérdidas de seguimiento; no acredita detección real."""
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from app.detector import NeuralDetector, Detection, ObjectTracker, crop_box, nms,analyze_objects
from app.localization import Region
from app.camera import CameraStream
from scripts.evaluate_detector import spatial_metrics
from tests.test_app import FakeCapture, FakeService, wait_until
from utils.config import PROJECT_ROOT


class DetectorTests(unittest.TestCase):
    def test_close_view_recovers_an_outline_around_a_partial_detection(self):
        detector=NeuralDetector.__new__(NeuralDetector)
        detector.confidence=.211
        frame=np.full((240,320,3),30,np.uint8)
        frame[40:190,80:230]=190
        partial=(106.,68.,195.,161.)
        base=Detection(partial,.42,Region(crop_box(partial,320,240),'neuronal'))
        with patch.object(detector,'detect',return_value=[base]):
            result=detector.detect_close(frame)
        self.assertEqual(result[0].confidence,base.confidence)
        np.testing.assert_allclose(result[0].box,[80,40,230,190],atol=2)
        with patch.object(detector,'detect',return_value=[]):
            self.assertEqual(detector.detect_close(frame),[])

    def test_close_view_rechecks_a_large_proposal_cut_by_the_frame(self):
        detector=NeuralDetector.__new__(NeuralDetector)
        detector.confidence=.211
        frame=np.zeros((240,320,3),np.uint8)
        first=Detection((0.,30.,250.,215.),.295,Region((0,20,250,225),'neural'))
        center=Detection((60.,30.,140.,110.),.26,Region((50,20,150,120),'neural'))
        with patch.object(detector,'detect',side_effect=[[first],[center]]) as detect:
            result=detector.detect_close(frame)
        self.assertEqual(detect.call_count,2)
        self.assertEqual(result[0].box,(105.,64.,185.,144.))
        self.assertEqual(result[0].confidence,first.confidence)

    def test_nms_and_letterbox_coordinates_outside_center(self):
        detector=NeuralDetector.__new__(NeuralDetector)
        from training.detector_own import ARCHITECTURE, STRIDES
        detector.architecture=ARCHITECTURE
        detector.size=512;detector.confidence=.35;detector.nms_iou=.45
        detector._lock=threading.Lock()
        # Frame 400x240: resize 512x307, top padding 102. Object at left.
        scale_x,scale_y=512/400,307/240
        x1,y1,x2,y2=20,30,80,180
        cx,cy=(x1+x2)/2*scale_x,(y1+y2)/2*scale_y+102
        boxes=torch.tensor([[x1*scale_x,y1*scale_y+102,x2*scale_x,y2*scale_y+102],
                            [x1*scale_x+1,y1*scale_y+103,x2*scale_x+1,y2*scale_y+103]])
        detector.model=lambda x:[]
        detector.model.strides=STRIDES
        with patch('app.detector.decode_predictions',return_value=[(boxes,torch.tensor([.9,.8]))]):
            detections=detector.detect(np.zeros((240,400,3),dtype=np.uint8))
        self.assertEqual(len(detections),1)
        np.testing.assert_allclose(detections[0].box,[x1,y1,x2,y2],atol=.1)
        self.assertLess(detections[0].region.box[0],x1)
        self.assertGreater(detections[0].region.box[2],x2)
        self.assertEqual(nms(np.array([[0,0,10,10],[1,1,11,11],[30,30,40,40]]),np.array([.9,.8,.7])),[0,2])

    def test_own_architecture_backward_and_official_checkpoints_rejected(self):
        from training.detector_own import VisionLocaliza,detection_loss
        import tempfile
        from pathlib import Path
        torch.set_num_threads(2)
        model=VisionLocaliza()
        self.assertEqual(sum(p.numel() for p in model.parameters()),630370)
        self.assertFalse(any('ultralytics' in type(m).__module__ or 'torchvision' in type(m).__module__ for m in model.modules()))
        output=model(torch.randn(2,3,128,128))
        self.assertEqual([tuple(o.shape) for o in output],[(2,6,16,16),(2,6,8,8),(2,6,4,4)])
        loss,_=detection_loss(output,[torch.tensor([[10.,20.,40.,90.]]),torch.tensor([[70.,20.,110.,90.]])])
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'official.pth'
            torch.save({'architecture':'YOLOv8n','provenance':{'initialization':'random','pretrained':False}},path)
            with self.assertRaisesRegex(ValueError,'arquitectura propia'):
                NeuralDetector(path)

    def test_empty_targets_have_finite_loss_and_targets_keep_displaced_objects(self):
        from training.detector_own import VisionLocaliza,detection_loss,make_targets
        torch.set_num_threads(2)
        model=VisionLocaliza()
        outputs=model(torch.randn(2,3,128,128))
        loss,parts=detection_loss(outputs,[torch.empty((0,4)),torch.empty((0,4))])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts['positive_cells'],0)
        targets,collisions=make_targets(outputs,[torch.tensor([[0.,10.,14.,30.]]),torch.tensor([[105.,90.,125.,120.]])])
        self.assertEqual(sum(int(t['positive'].sum()) for t in targets),2)
        self.assertEqual(collisions,0)
        self.assertTrue(targets[0]['positive'][0,:,0].any())
        self.assertTrue(targets[0]['positive'][1,:,-2:].any())

    def test_closeup_augmentation_transforms_real_boxes_without_full_image_labels(self):
        from training.detection_data import annotated_crop
        frame=np.zeros((200,300,3),np.uint8)
        boxes=np.array([[100.,40.,140.,160.],[260.,10.,280.,25.]],dtype=np.float32)
        cfg={'object_crop_probability':1.,'background_crop_probability':0.,'focus_categories':['bottle']}
        with patch('training.detection_data.random.random',return_value=.5),patch('training.detection_data.random.uniform',return_value=.5):
            image,transformed=annotated_crop(frame,boxes,['bottle','other'],cfg)
        self.assertEqual(image.shape,(200,80,3))
        np.testing.assert_array_equal(transformed,[[20.,40.,60.,160.]])
        self.assertNotEqual(transformed[0].tolist(),[0,0,80,200])

    def test_resume_dataset_extension_rejects_modified_holdout_or_non_train_source(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from scripts.train_detector_own import audited_training_extension
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=[{'id':1,'split':'train'},{'id':2,'split':'val'},{'id':3,'split':'test'}]
            parent=json.dumps(original).encode()
            previous=b'{"original":true}'
            previous_sha=hashlib.sha256(previous).hexdigest()
            (root/'parent_audit.json').write_bytes(previous)
            (root/'parent_manifest.json').write_bytes(parent)
            (root/'audit.json').write_text(json.dumps({'parent_audit_sha256':previous_sha,
                'parent_manifest_sha256':hashlib.sha256(parent).hexdigest(),'cross_split_candidates':0,
                'foundation_annotations_used':False,'source':'official human annotations'}))
            extra={'id':300001,'split':'train','source_original_split':'train',
                   'source_dataset':'MJU-Waste','human_polygon_mask_IoU':.98}
            (root/'manifest.json').write_text(json.dumps(original+[extra]))
            self.assertEqual(audited_training_extension(root,previous_sha)['added_train_images'],1)
            altered=[dict(r) for r in original];altered[1]['split']='train'
            (root/'manifest.json').write_text(json.dumps(altered+[extra]))
            with self.assertRaises(ValueError):audited_training_extension(root,previous_sha)
            extra['source_original_split']='test'
            (root/'manifest.json').write_text(json.dumps(original+[extra]))
            with self.assertRaises(ValueError):audited_training_extension(root,previous_sha)

    def test_ap_counts_spatial_misses_and_false_positives(self):
        truth=[{'box':[0,0,10,10],'category':'Glass bottle'}]
        good={'box':[0,0,10,10],'score':.9}
        false={'box':[20,20,30,30],'score':.95}
        perfect=spatial_metrics([{'truth':truth,'predictions':[good]}])
        self.assertAlmostEqual(perfect['mAP50_95'],1.)
        mixed=spatial_metrics([{'truth':truth,'predictions':[good,false]},{'truth':truth,'predictions':[]}])
        self.assertEqual(mixed['precision'],.5)
        self.assertEqual(mixed['recall'],.5)
        self.assertLess(mixed['mAP50'],.3)

    def test_operating_point_uses_validation_and_preserves_classifier(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from scripts.select_detector import select
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'own.pth';checkpoint.write_bytes(b'own weights fingerprint')
            sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            app=root/'app.json';app.write_text(json.dumps({'checkpoint':'classifier_B.pth'}))
            truth=[{'box':[0,0,10,10],'category':'residuo'}]
            good={'box':[0,0,10,10],'score':.8}
            false={'box':[30,30,40,40],'score':.9}
            candidates=[]
            for name,nms,predictions in [('duplicate',.45,[good,false]),('clean',.25,[good])]:
                folder=root/name;folder.mkdir();candidates.append(folder)
                (folder/'metrics.json').write_text(json.dumps({'split':'val','checkpoint_sha256':sha,'nms_iou':nms}))
                (folder/'predictions.json').write_text(json.dumps([{'image_id':1,'truth':truth,'predictions':predictions}]))
            frozen=select(checkpoint,candidates,app)
            self.assertEqual(frozen['nms_iou'],.25)
            self.assertEqual(frozen['operating_point']['F1'],1.)
            self.assertFalse(frozen['test_used'])
            self.assertEqual(json.loads(app.read_text())['checkpoint'],'classifier_B.pth')
            (candidates[0]/'metrics.json').write_text(json.dumps({'split':'test','checkpoint_sha256':sha,'nms_iou':.45}))
            with self.assertRaisesRegex(ValueError,'validación'):select(checkpoint,candidates,app)

    def test_loss_removes_tracks_and_does_not_classify_empty_frame(self):
        tracker=ObjectTracker();service=FakeService()
        frame=np.zeros((100,120,3),np.uint8)
        d=Detection((10,10,50,60),.9,Region((5,5,65,65),'neuronal'))
        first=tracker.update(frame,[d],service)
        second=tracker.update(frame,[d],service)
        self.assertEqual(first[0].track_id,second[0].track_id)
        self.assertTrue(second[0].confirmed)
        self.assertEqual(tracker.update(frame,[],service),[])
        self.assertEqual(service.calls,2)
        self.assertEqual(tracker.tracks,{})

    def test_one_residue_is_selected_and_does_not_jump_for_small_score_changes(self):
        frame=np.zeros((120,240,3),np.uint8)
        service=FakeService();tracker=ObjectTracker()
        a=Detection((10,10,60,100),.85,Region((5,5,65,105),'neuronal'))
        b=Detection((150,10,200,100),.80,Region((145,5,205,105),'neuronal'))
        first=tracker.update(frame,[b,a],service)
        self.assertEqual(len(first),1)
        self.assertEqual(first[0].detection.box,a.box)
        from dataclasses import replace
        second=tracker.update(frame,[replace(b,confidence=.90),a],service)
        self.assertEqual(second[0].track_id,first[0].track_id)
        self.assertEqual(second[0].detection.box,a.box)
        third=tracker.update(frame,[replace(b,confidence=.98),replace(a,confidence=.50)],service)
        self.assertEqual(third[0].detection.box,b.box)
        self.assertNotEqual(third[0].track_id,first[0].track_id)
        self.assertEqual(service.calls,3)
        self.assertEqual(tracker.update(frame,[],service),[])
        self.assertEqual(service.calls,3)

    def test_photo_classifies_only_best_detection_and_empty_has_no_CNN_call(self):
        frame=np.zeros((120,240,3),np.uint8);service=FakeService()
        a=Detection((10,10,60,100),.7,Region((5,5,65,105),'neuronal'))
        b=Detection((150,10,200,100),.9,Region((145,5,205,105),'neuronal'))
        class Detector:
            def detect(self,frame):return [a,b]
        objects=analyze_objects(frame,Detector(),service)
        self.assertEqual(len(objects),1)
        self.assertEqual(objects[0].detection.box,b.box)
        self.assertEqual(service.calls,1)
        with patch.object(Detector,'detect',return_value=[]):
            self.assertEqual(analyze_objects(frame,Detector(),service),[])
        self.assertEqual(service.calls,1)

    def test_neural_camera_needs_no_background_and_clears_loss(self):
        class Detector:
            active=True
            def detect(self,frame):
                return [Detection((10,10,50,60),.9,Region((5,5,65,65),'neuronal'))] if self.active else []
        detector=Detector();service=FakeService()
        stream=CameraStream(service,mode='neural',detector=detector,capture_factory=lambda i:FakeCapture(),interval=.01).start()
        try:
            self.assertTrue(wait_until(lambda:bool(stream.snapshot().objects)))
            snap=stream.snapshot()
            self.assertFalse(snap.background_ready)
            np.testing.assert_array_equal(snap.frame,snap.analysis_frame)
            detector.active=False
            self.assertTrue(wait_until(lambda:not stream.snapshot().objects))
            self.assertIsNone(stream.snapshot().reading)
        finally:stream.stop()

    def test_neural_ui_no_detections_does_not_invoke_CNN(self):
        import io
        from PIL import Image
        class Detector:
            def detect(self,frame):return []
        payload=io.BytesIO();Image.new('RGB',(200,200)).save(payload,'PNG')
        with patch('streamlit.file_uploader',return_value=payload),patch('app.resources.get_detector',return_value=Detector()):
            app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
            app.radio[0].set_value('Subir imagen').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No se reconoci' in i.value for i in app.info))

    def test_slow_detector_does_not_block_live_capture_or_republish_after_stop(self):
        import time
        gate=threading.Event();entered=threading.Event()
        class Capture(FakeCapture):
            number=0
            def read(self):
                self.number+=1
                return True,np.full((100,120,3),self.number%250,np.uint8)
        class SlowDetector:
            def detect(self,frame):
                entered.set();gate.wait(2)
                return []
        capture=Capture()
        stream=CameraStream(FakeService(),mode='neural',detector=SlowDetector(),capture_factory=lambda _:capture).start()
        try:
            self.assertTrue(entered.wait(1))
            first=stream.snapshot()
            self.assertIsNotNone(first.frame)
            self.assertIsNone(first.analysis_frame)
            self.assertTrue(wait_until(lambda:not np.array_equal(first.frame,stream.snapshot().frame)))
            stream.stop();gate.set()
            self.assertTrue(wait_until(lambda:capture.released))
            self.assertIsNone(stream.snapshot().frame)
            self.assertIsNone(stream.snapshot().reading)
        finally:gate.set();stream.stop()

    def test_preview_uses_newest_capture_and_does_not_draw_stale_boxes(self):
        import time
        from app.camera import Snapshot
        from app.live_view import preview_frame
        frame=np.full((100,120,3),170,np.uint8)
        stale=Snapshot('running',frame,None,time.monotonic()-2,'',np.zeros_like(frame),objects=(object(),))
        np.testing.assert_array_equal(preview_frame(stale),frame)
        fresh=Snapshot('running',frame,None,time.monotonic(),'',np.zeros_like(frame),objects=(object(),))
        with patch('app.live_view.annotate_objects',return_value=frame) as annotate:
            preview_frame(fresh)
            self.assertIs(annotate.call_args.args[0],frame)



if __name__=='__main__':unittest.main()
