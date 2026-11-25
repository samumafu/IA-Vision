import unittest

import torch
from torch import nn

from training.material_encoder import MaterialEncoder


class MaterialEncoderTests(unittest.TestCase):
    def test_encoder_layers_initialize_independently_and_learn(self):
        torch.set_num_threads(2)
        model = MaterialEncoder({'image_size': 160, 'width': 32, 'heads': 4, 'depth': 2, 'dropout': .1})
        self.assertIsInstance(model.encoder, nn.TransformerEncoder)
        first, second = model.encoder.layers
        self.assertFalse(torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight))
        output = model(torch.randn(2, 3, 160, 160))
        self.assertEqual(output.shape, (2, 6))
        output.square().mean().backward()
        for layer in (first, second):
            gradient = layer.self_attn.in_proj_weight.grad
            self.assertTrue(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)

    def test_convolution_only_ablation_has_same_tokenizer(self):
        config = {'image_size': 160, 'width': 32, 'heads': 4, 'depth': 0, 'dropout': .1}
        torch.manual_seed(2)
        cnn = MaterialEncoder(config)
        torch.manual_seed(2)
        encoder = MaterialEncoder(config | {'depth': 2})
        self.assertIsInstance(cnn.encoder, nn.Identity)
        torch.testing.assert_close(cnn.project.weight, encoder.project.weight)
