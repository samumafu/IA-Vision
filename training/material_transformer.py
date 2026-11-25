"""VisionMaterial-Transformer, a compact visual classifier with learned readout tokens."""

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'VisionMaterial-Transformer'
IMAGE_SIZES = (128, 160, 192)


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.feed_forward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, tokens):
        batch, count, width = tokens.shape
        query, key, value = self.qkv(self.attention_norm(tokens)).reshape(
            batch, count, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.)
        tokens = tokens + self.projection(attended.transpose(1, 2).reshape(batch, count, width))
        return tokens + self.feed_forward(tokens)


class MaterialTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config['image_size']
        width, heads, depth = (config[key] for key in ('width', 'heads', 'depth'))
        dropout = config['dropout']
        if (self.image_size not in IMAGE_SIZES or any(type(v) is not int or v <= 0 for v in (width, heads, depth))
                or width % heads):
            raise ValueError('Invalid transformer size, width, heads, or depth.')
        self.texture = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 40, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(40), nn.GELU(),
            nn.Conv2d(40, 56, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(56), nn.GELU(),
        )
        self.token_projection = nn.Conv2d(112, width, 1)
        grid = self.image_size // 16
        self.rows = nn.Parameter(torch.empty(1, grid, 1, width))
        self.columns = nn.Parameter(torch.empty(1, 1, grid, width))
        self.readout = nn.Parameter(torch.empty(1, 2, width))
        for parameter in (self.rows, self.columns, self.readout):
            nn.init.trunc_normal_(parameter, std=.02)
        self.blocks = nn.Sequential(*(AttentionBlock(width, heads, dropout) for _ in range(depth)))
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Dropout(dropout), nn.Linear(2 * width, 6))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f'Expected RGB crops of size {self.image_size} × {self.image_size}.')
        local = self.texture(images)
        patches = self.token_projection(torch.cat((F.avg_pool2d(local, 2), F.max_pool2d(local, 2)), dim=1))
        patches = patches.permute(0, 2, 3, 1)
        patches = (patches + self.rows.to(patches.dtype) + self.columns.to(patches.dtype)).flatten(1, 2)
        readout = self.readout.to(patches.dtype).expand(images.shape[0], -1, -1)
        tokens = self.blocks(torch.cat((readout, patches), dim=1))
        return self.head(tokens[:, :2].flatten(1))
