"""Configuracion, semillas y checkpoints propios portables entre CUDA y CPU."""
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from training.data import resolve_path
from training.model import ARCHITECTURE, VisionCNN
from training.material_encoder import ARCHITECTURE as ENCODER_ARCHITECTURE, IMAGE_SIZES as ENCODER_SIZES, MaterialEncoder
from training.material_transformer import ARCHITECTURE as TRANSFORMER_ARCHITECTURE, IMAGE_SIZES, MaterialTransformer
from utils.config import CLASSES, PROJECT_ROOT


def load_config(path=None, overrides=None):
    path = resolve_path(path or "config/train_A.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if config["image_size"] != 128:
        raise ValueError("La CNN definida requiere image_size=128.")
    for key in ("epochs", "batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} debe ser un entero positivo.")
    if config["num_workers"] < 0 or config["learning_rate"] <= 0 or not math.isfinite(config["learning_rate"]):
        raise ValueError("num_workers o learning_rate invalidos.")
    if not 0 <= config["dropout"] < 1 or config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("dropout o device invalidos.")
    early = config["early_stopping"]
    if not isinstance(early["patience"], int) or early["patience"] < 1 or early["min_delta"] < 0:
        raise ValueError("Configuracion de early stopping invalida.")
    norm = config["normalization"]
    if len(norm["mean"]) != 3 or len(norm["std"]) != 3 or any(s <= 0 for s in norm["std"]):
        raise ValueError("Normalizacion RGB invalida.")
    augmentation = config["augmentation"]
    if not 0 <= augmentation["horizontal_flip_probability"] <= 1 or not 0 <= augmentation["rotation_degrees"] <= 30:
        raise ValueError("Aumentacion geometrica fuera de los limites moderados.")
    if any(not 0 <= augmentation[key] <= .3 for key in ("brightness", "contrast")):
        raise ValueError("Brillo/contraste deben estar entre 0 y 0.3.")
    return config


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA AdaptiveAvgPool2d no tiene backward determinista en algunas versiones.
    # Mantener semillas y preferir determinismo, permitiendo avisos en esas operaciones.
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(value="auto"):
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero PyTorch no detecta una GPU NVIDIA compatible.")
    return torch.device("cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available()) else "cpu")


class EarlyStopping:
    def __init__(self, patience=6, min_delta=.001):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = -float("inf"), 0

    def step(self, score):
        if not math.isfinite(score):
            raise ValueError("Macro F1 no finito.")
        if score > self.best + self.min_delta:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(path, model, config, epoch, val_macro_f1, manifest_sha256):
    # Solo tensores y tipos simples: se puede leer con weights_only=True en CPU.
    checkpoint = {
        "format_version": 1, "architecture": ARCHITECTURE, "classes": list(CLASSES),
        "image_size": 128, "config": config, "epoch": int(epoch),
        "val_macro_f1": float(val_macro_f1), "manifest_sha256": manifest_sha256,
        "torch_version": str(torch.__version__),
        "model_state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Se carga un checkpoint propio explicito, nunca al construir VisionCNN.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el checkpoint: {path}. Copie best_model.pth a models/ o indique --checkpoint RUTA.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"architecture", "classes", "image_size", "config", "model_state_dict", "manifest_sha256"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Se necesita el checkpoint completo de Vision-IA, con etiquetas y normalizacion originales; no basta un state_dict sin metadatos.")
    architecture, image_size = checkpoint["architecture"], checkpoint["image_size"]
    if not ((architecture == ARCHITECTURE and image_size == 128) or
            (architecture == "VisionResidual-v2" and image_size in (160, 192)) or
            (architecture == TRANSFORMER_ARCHITECTURE and image_size in IMAGE_SIZES) or
            (architecture == ENCODER_ARCHITECTURE and image_size in ENCODER_SIZES)):
        raise ValueError("Arquitectura o resolución de checkpoint incompatible.")
    if checkpoint["classes"] != list(CLASSES):
        raise ValueError(f"Orden de clases incompatible. Esperado: {list(CLASSES)}; checkpoint: {checkpoint['classes']}. No se reasignan etiquetas automaticamente.")
    if "class_to_idx" in checkpoint and checkpoint["class_to_idx"] != {c: i for i, c in enumerate(CLASSES)}:
        raise ValueError("class_to_idx no coincide con el orden original del checkpoint.")
    config = checkpoint["config"]
    if not isinstance(config, dict) or config.get("image_size") != image_size or "normalization" not in config:
        raise ValueError("El checkpoint no conserva la configuracion de entrada original.")
    if architecture != ARCHITECTURE and (config.get("architecture") != architecture or config.get("preprocessing") not in ("letterbox-v2","resize-v2")):
        raise ValueError("Faltan la arquitectura y el preprocesamiento originales del modelo.")
    norm = config["normalization"]
    if not isinstance(norm, dict) or any(not isinstance(norm.get(key), (list, tuple)) or len(norm[key]) != 3 for key in ("mean", "std")):
        raise ValueError("Faltan media y desviacion RGB originales del checkpoint.")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for key in ("mean", "std") for v in norm[key]) or any(v <= 0 for v in norm["std"]):
        raise ValueError("Normalizacion del checkpoint invalida.")
    dropout = config.get("dropout")
    if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("Dropout del checkpoint invalido.")
    if architecture == "VisionResidual-v2":
        from training.compact import CompactCNN
        model = CompactCNN(image_size, dropout, config.get("channels",(24,48,96,160)), config.get("pooling","avg"))
    elif architecture in (TRANSFORMER_ARCHITECTURE, ENCODER_ARCHITECTURE):
        provenance = checkpoint.get('provenance', {})
        if provenance.get('initialization') != 'random_from_scratch' or provenance.get('pretrained') is not False:
            raise ValueError('El Transformer requiere un checkpoint propio entrenado desde cero.')
        model = MaterialTransformer(config) if architecture == TRANSFORMER_ARCHITECTURE else MaterialEncoder(config)
    else:
        model = VisionCNN(dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
