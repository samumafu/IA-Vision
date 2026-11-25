import unittest
from unittest.mock import Mock, patch
import time

import torch

from app.localization import Region
from app.vision import InferenceService, Reading
from training.material_transformer import MaterialTransformer
from utils.config import CLASSES


class MaterialTransformerTests(unittest.TestCase):
    def test_readout_depends_on_images_through_attention(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)
        model = MaterialTransformer({'image_size': 128, 'width': 32, 'heads': 4, 'depth': 2, 'dropout': 0.}).eval()
        batch = torch.stack((torch.zeros(3, 128, 128), torch.randn(3, 128, 128)))
        with torch.inference_mode():
            logits = model(batch)
            with patch('training.material_transformer.F.scaled_dot_product_attention',
                       side_effect=lambda query, *args, **kwargs: torch.zeros_like(query)) as attention:
                isolated = model(batch)
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(attention.call_count, 2)
        self.assertFalse(torch.allclose(logits[0], logits[1]))
        torch.testing.assert_close(isolated[0], isolated[1])

    def test_fusion_refines_same_region_and_preserves_protected_material(self):
        service = InferenceService.__new__(InferenceService)
        service.classes = CLASSES
        service.refiner = Mock()
        service.refiner_weight = .55
        service.refiner_preserve = frozenset()
        service.refiner_preserve_consensus = frozenset({'plastico'})
        service.refiner_calibration = None
        region = Region((0, 0, 32, 32), 'test')
        base = Reading('plastico', .42, dict(zip(CLASSES, (.42, .38, .08, .04, .04, .04))), region, 4.)
        other = Reading('papel', .8, dict(zip(CLASSES, (.04, .8, .04, .04, .04, .04))), region, 3.)
        service.refiner.classify.return_value = other
        frame = object()
        improved = service._refine(frame, base, time.perf_counter())
        service.refiner.classify.assert_called_once_with(frame, region=region)
        self.assertEqual(improved.material, 'papel')
        self.assertEqual(improved.views, 2)
        self.assertAlmostEqual(sum(improved.probabilities.values()), 1.)
        service.refiner.reset_mock()
        self.assertIs(service._refine(frame, base, time.perf_counter(), agreement='plastico'), base)
        service.refiner_preserve = frozenset({'plastico'})
        self.assertIs(service._refine(frame, base, time.perf_counter()), base)
        service.refiner.classify.assert_not_called()
