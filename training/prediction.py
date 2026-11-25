"""Inferencia con etiquetas/normalizacion del checkpoint propio, sin entrenamiento."""
import csv
from pathlib import Path

import torch
from PIL import Image

from training.compact import ARCHITECTURE as RESIDUAL_ARCHITECTURE, build_transform
from training.data import ImageTransform
from training.material_encoder import ARCHITECTURE as ENCODER_ARCHITECTURE
from training.material_transformer import ARCHITECTURE as TRANSFORMER_ARCHITECTURE
from training.runtime import choose_device, load_checkpoint
from utils.config import PROJECT_ROOT, data_root
from utils.dataset import sha256


def probabilities_from_logits(logits):
    if logits.ndim != 2 or logits.shape[1] != 6 or not torch.isfinite(logits).all():
        raise ValueError("El clasificador debe producir seis logits finitos por imagen.")
    return torch.softmax(logits, dim=1)


def model_probabilities(model, images, config):
    probabilities = probabilities_from_logits(model(images))
    if config.get("inference_hflip", False):
        probabilities = (probabilities + probabilities_from_logits(model(images.flip(-1))))/2
    return probabilities


def protect_test_image(path):
    """No abre fotos test: comprueba rutas y hashes ya registrados en manifiestos."""
    roots = {base / "experiments" / experiment / "processed"
             for base in (PROJECT_ROOT / "data", data_root()) for experiment in ("A", "B")}
    path = Path(path).resolve()
    digest = None
    for root in roots:
        if path.is_relative_to((root / "test").resolve()):
            raise ValueError("La imagen pertenece a PRUEBA. Ese conjunto permanece reservado; utilice una fotografia nueva.")
        manifest = root / "manifest.csv"
        if manifest.exists():
            if digest is None:
                digest = sha256(path)
            with manifest.open(encoding="utf-8", newline="") as stream:
                if any(row["split"] == "test" and row["sha256"] == digest for row in csv.DictReader(stream)):
                    raise ValueError("La imagen coincide con un archivo de PRUEBA, aunque se haya copiado a otra carpeta.")


def predict_image(checkpoint_path, image_path, device="cpu"):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"No se encontro la fotografia: {image_path}")
    protect_test_image(image_path)
    device = choose_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint['architecture'] in (RESIDUAL_ARCHITECTURE, TRANSFORMER_ARCHITECTURE, ENCODER_ARCHITECTURE):
        transform = build_transform(checkpoint['config'], training=False)
    else:
        transform = ImageTransform(training=False, normalization=checkpoint['config']['normalization'])
    with Image.open(image_path) as image:
        tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model_probabilities(model, tensor, checkpoint["config"])[0].cpu().tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    classes = checkpoint["classes"]
    return {
        "image": str(image_path), "predicted_class": classes[index], "confidence": probabilities[index],
        "probabilities": dict(zip(classes, probabilities)), "classes": classes,
        "normalization": checkpoint["config"]["normalization"],
        "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
        "confidence_definition": "Probabilidad softmax maxima; no es una probabilidad calibrada de acierto.",
    }
