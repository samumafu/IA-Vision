"""La candidata dudosa conserva la respuesta de B y cuenta ambas inferencias."""
from dataclasses import replace
from unittest.mock import Mock,patch
import unittest

import numpy as np

from app.vision import InferenceService


class MaterialGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service=InferenceService('models/runtime/classifier_B.pth')
        cls.frame=np.full((48,64,3),128,dtype=np.uint8)
        cls.original=cls.service.classify(cls.frame)

    def tearDown(self):self.service.candidate=None

    def test_uncertain_candidate_keeps_original_and_counts_both_views(self):
        candidate=replace(self.original,material='plastico' if self.original.material!='plastico' else 'metal',confidence=.39)
        self.service.candidate=Mock();self.service.candidate.classify.return_value=candidate
        result=self.service.classify(self.frame)
        self.assertEqual(result.material,self.original.material)
        self.assertEqual(result.confidence,self.original.confidence)
        self.assertEqual(result.views,self.original.views+candidate.views)

    def test_sufficient_candidate_avoids_the_fallback_forward(self):
        candidate=replace(self.original,confidence=.8)
        self.service.candidate=Mock();self.service.candidate.classify.return_value=candidate
        with patch.object(self.service.model,'forward',side_effect=AssertionError('Fallback should not run')):
            self.assertIs(self.service.classify(self.frame),candidate)

    def test_weighted_policy_uses_both_distributions_even_at_low_confidence(self):
        refined = 'models/runtime/classifier_materials.pth'
        service = InferenceService(refined, candidate_checkpoint='models/runtime/classifier_B.pth',
                                   candidate_threshold=1., candidate_weight=.55)
        primary = InferenceService(refined).classify(self.frame)
        blended = service.classify(self.frame)
        for label in service.classes:
            expected = .45 * primary.probabilities[label] + .55 * self.original.probabilities[label]
            self.assertAlmostEqual(blended.probabilities[label], expected)
        self.assertEqual(blended.views, primary.views + self.original.views)


if __name__=='__main__':unittest.main()
