"""CNN residual pequeña y transformación RGB propia. No importa pesos ni torchvision."""
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn

ARCHITECTURE = "VisionResidual-v2"


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False), nn.BatchNorm2d(outgoing))
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images):
        return self.activation(self.path(images) + self.skip(images))


class CompactCNN(nn.Module):
    def __init__(self, image_size=192, dropout=.25, channels=(24,48,96,160), pooling="avg"):
        super().__init__()
        if image_size not in (160, 192):
            raise ValueError("VisionResidual-v2 admite 160 o 192 píxeles.")
        self.image_size = image_size
        if len(channels)!=4 or any(not isinstance(value,int) or not 16<=value<=512 for value in channels) or pooling not in ("avg","avgmax"):
            raise ValueError("Configuración residual incompatible.")
        self.pooling=pooling
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 5, 2, 2, bias=False), nn.BatchNorm2d(channels[0]),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        layers, incoming = [], channels[0]
        for stage, outgoing in enumerate(channels):
            layers += [ResidualBlock(incoming, outgoing, 1 if stage == 0 else 2), ResidualBlock(outgoing, outgoing)]
            incoming = outgoing
        self.blocks = nn.Sequential(*layers)
        dimension=channels[-1]*(2 if pooling=="avgmax" else 1)
        self.head = nn.Sequential(nn.LayerNorm(dimension),nn.Dropout(dropout),nn.Linear(dimension,6)) if pooling=="avgmax" else nn.Sequential(nn.Dropout(dropout),nn.Linear(dimension,6))
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"Se requiere RGB {self.image_size} × {self.image_size}.")
        features=self.blocks(self.stem(images))
        pooled=features.mean(dim=(2,3))
        if self.pooling=="avgmax":pooled=torch.cat((pooled,features.amax(dim=(2,3))),dim=1)
        return self.head(pooled)


class CompactTransform:
    """Letterbox conserva el objeto completo; solo train aplica variaciones aleatorias."""
    def __init__(self, config, training=False):
        self.size, self.training = config["image_size"], training
        self.preprocessing=config.get("preprocessing","letterbox-v2")
        self.inference_scale=float(config.get("inference_scale",1.))
        if not .8<=self.inference_scale<=1.:
            raise ValueError("La escala determinista de entrada debe estar entre 0.8 y 1.")
        if self.preprocessing not in ("letterbox-v2","resize-v2"):
            raise ValueError("Preprocesamiento desconocido.")
        self.options = config.get("augmentation", {})
        self.normalization = config["normalization"]
        self.mean = torch.tensor(self.normalization["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(self.normalization["std"], dtype=torch.float32)[:, None, None]

    def prepare_image(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = self.size
        if not self.training:
            if self.preprocessing=="resize-v2":
                side=round(size*self.inference_scale)
                image=image.resize((side,side),Image.Resampling.BILINEAR)
                if side==size:return image
                canvas=Image.new("RGB",(size,size),(128,128,128))
                canvas.paste(image,((size-side)//2,)*2)
                return canvas
            return ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(128, 128, 128))
        working = round(size*self.options.get("working_scale", 1.25))
        image = image.resize((working,working),Image.Resampling.BILINEAR) if self.preprocessing=="resize-v2" else ImageOps.contain(image, (working, working), method=Image.Resampling.BILINEAR)
        if random.random() < .5:
            image = ImageOps.mirror(image)
        angle = self.options.get("rotation_degrees", 20)
        clean_layout=self.options.get("layout")=="clean"
        camera_layout = self.options.get("layout") == "camera"
        image = image.rotate(random.uniform(-angle, angle), Image.Resampling.BILINEAR,
                             expand=not (clean_layout or camera_layout), fillcolor=(128, 128, 128))
        for option, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast), ("color", ImageEnhance.Color)):
            amount = self.options.get(option, .25)
            image = enhancer(image).enhance(random.uniform(1-amount, 1+amount))
        if random.random() < .08:
            image = ImageOps.grayscale(image).convert("RGB")
        if random.random() < .1:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(.2, .8)))
        if clean_layout and random.random()>=self.options.get("scale_probability",.2):
            canvas=image.resize((size,size),Image.Resampling.BILINEAR)
        elif camera_layout and random.random() >= self.options.get("scale_probability", .2):
            canvas = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(128, 128, 128))
        else:
            factor = random.uniform(self.options.get("min_scale", .8), 1.)
            image = ImageOps.contain(image, (max(16, round(size*factor)),)*2, method=Image.Resampling.BILINEAR)
            gray = random.randint(100, 155)
            canvas = Image.new("RGB", (size, size), (gray, gray, gray))
            canvas.paste(image, (random.randint(0, size-image.width), random.randint(0, size-image.height)))
        gain=self.options.get("channel_gain",0.)
        if gain:
            pixels=np.asarray(canvas,dtype=np.float32)*np.random.uniform(1-gain,1+gain,(1,1,3))
            canvas=Image.fromarray(np.clip(pixels,0,255).astype(np.uint8))
        return canvas

    def __call__(self, image):
        pixels = np.asarray(self.prepare_image(image), dtype=np.float32).transpose(2, 0, 1).copy()/255.
        return (torch.from_numpy(pixels)-self.mean)/self.std


def build_transform(config, training=False):
    from training.material_encoder import ARCHITECTURE as ENCODER_ARCHITECTURE
    from training.material_transformer import ARCHITECTURE as TRANSFORMER_ARCHITECTURE
    if config.get("architecture") in (ARCHITECTURE, TRANSFORMER_ARCHITECTURE, ENCODER_ARCHITECTURE):
        return CompactTransform(config, training)
    from training.data import ImageTransform
    return ImageTransform(training, config["normalization"], config.get("augmentation"))
