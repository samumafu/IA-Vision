"""VisionMaterial-Encoder uses an original CNN tokenizer and PyTorch attention."""

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'VisionMaterial-Encoder'
IMAGE_SIZES = (160, 192)


def convolution(incoming, outgoing, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride, 1, groups=groups, bias=False),
                         nn.BatchNorm2d(outgoing), nn.SiLU(inplace=True))


class TextureResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = convolution(channels, channels, groups=channels)
        self.mix = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, image):
        return F.silu(image + self.mix(self.spatial(image)))


def texture_stage(incoming, outgoing, blocks):
    return nn.Sequential(convolution(incoming, outgoing, stride=2), *(TextureResidual(outgoing) for _ in range(blocks)))


class MaterialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config['image_size']
        depth, width, heads = (config[key] for key in ('depth', 'width', 'heads'))
        dropout = config['dropout']
        if (self.image_size not in IMAGE_SIZES or any(type(v) is not int for v in (depth, width, heads))
                or depth < 0 or width < 1 or heads < 1 or width % heads):
            raise ValueError('Invalid convolutional encoder configuration.')
        self.stem = nn.Sequential(convolution(3, 24, stride=2), convolution(24, 24))
        self.fine = texture_stage(24, 32, 1)
        self.middle = texture_stage(32, 56, 2)
        self.coarse = texture_stage(56, 88, 2)
        self.project = nn.Conv2d(56 + 88, width, 1)
        grid = self.image_size // 16
        self.position = nn.Parameter(torch.empty(1, grid * grid, width)) if depth else None
        if depth:
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * 2,
                                               dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
            nn.init.trunc_normal_(self.position, std=.02)
            # TransformerEncoder clones its template. Reinitialize each layer independently.
            for cloned in self.encoder.layers:
                nn.init.xavier_uniform_(cloned.self_attn.in_proj_weight)
                for linear in (cloned.self_attn.out_proj, cloned.linear1, cloned.linear2):
                    nn.init.xavier_uniform_(linear.weight)
        else:
            self.encoder = nn.Identity()
        self.classifier = nn.Sequential(nn.LayerNorm(3 * width), nn.Dropout(dropout), nn.Linear(3 * width, 6))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f'Expected RGB crops of {self.image_size} × {self.image_size} pixels.')
        middle = self.middle(self.fine(self.stem(images)))
        coarse = self.coarse(middle)
        features = self.project(torch.cat((F.avg_pool2d(middle, 2), coarse), dim=1))
        local = features.flatten(2).transpose(1, 2)
        tokens = local if self.position is None else local + self.position.to(local.dtype)
        contextual = self.encoder(tokens)
        summary = torch.cat((local.amax(dim=1), contextual.mean(dim=1), contextual.amax(dim=1)), dim=1)
        return self.classifier(summary)
