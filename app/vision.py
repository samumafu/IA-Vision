"""CNN propia y recortes explícitos; el detector neuronal vive en app.detector."""
import hashlib
import io
import json
from pathlib import Path
import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from training.compact import build_transform
from training.prediction import model_probabilities
from training.runtime import load_checkpoint
from utils.dataset import sha256
from app.localization import Region, Localizer, LocalizationError


@dataclass(frozen=True)
class Reading:
    material: str
    confidence: float
    probabilities: dict
    region: Region
    inference_ms: float
    crop_sha256: str = ""
    tensor_sha256: str = ""
    image_size: int = 128
    views: int = 1


def decode_image(payload, max_bytes=10 * 1024 * 1024, max_pixels=16000000, reserved_hashes=frozenset()):
    if not payload or len(payload) > max_bytes:
        raise ValueError("El archivo está vacío o supera el tamaño permitido.")
    if hashlib.sha256(payload).hexdigest() in reserved_hashes:
        raise ValueError("Esta imagen pertenece al conjunto de prueba reservado. Usa una fotografía nueva.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > max_pixels:
                raise ValueError("La imagen es demasiado grande. Reduce su resolución antes de cargarla.")
            source.verify()
        with Image.open(io.BytesIO(payload)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if min(image.size) < 32:
                raise ValueError("La imagen debe medir al menos 32 píxeles por lado.")
            return np.asarray(image).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("No se pudo leer la imagen. Prueba con un JPG, PNG o WebP válido.") from exc


def central_region(rgb, fraction=.72):
    height, width = rgb.shape[:2]
    side = max(1, int(min(height, width) * fraction))
    left, top = (width - side) // 2, (height - side) // 2
    return Region((left, top, left + side, top + side), "central")


def find_region(rgb, fraction=.72, use_contours=False):
    if not use_contours:
        return central_region(rgb, fraction)  # Compatibilidad explícita; la UI nueva no lo utiliza.
    return Localizer("auto").locate(rgb).region


def bgr_to_rgb(bgr):
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Fotograma BGR inválido.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_region(rgb, region):
    left, top, right, bottom = region.box
    height, width = rgb.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height and min(right-left, bottom-top) >= 16):
        raise ValueError("Recorte fuera de límites o demasiado pequeño.")
    return rgb[top:bottom, left:right].copy()


def annotate_region(rgb, region):
    output = rgb.copy()
    x1, y1, x2, y2 = region.box
    color, thickness = (21, 126, 117), max(2, min(rgb.shape[:2]) // 150)
    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
    length = max(12, (x2 - x1) // 12)
    for x, direction_x in ((x1, 1), (x2 - 1, -1)):
        for y, direction_y in ((y1, 1), (y2 - 1, -1)):
            cv2.line(output, (x, y), (x + length * direction_x, y), (255, 255, 255), thickness + 1)
            cv2.line(output, (x, y), (x, y + length * direction_y), (255, 255, 255), thickness + 1)
    return output


class InferenceService:
    def __init__(self, checkpoint, cpu_threads=2, candidate_checkpoint=None, candidate_threshold=.4,
                 candidate_weight=None, refiner_checkpoint=None, refiner_weight=None,
                 refiner_preserve=(), refiner_calibration=None, refiner_preserve_consensus=()):
        if not 1 <= cpu_threads <= 4:
            raise ValueError("Usa entre uno y cuatro hilos de CPU.")
        torch.set_num_threads(cpu_threads)
        self.cpu_threads=cpu_threads
        cv2.setNumThreads(1)
        self.model, self.metadata = load_checkpoint(checkpoint, "cpu")
        self.classes = tuple(self.metadata["classes"])
        self.transform = build_transform(self.metadata["config"], training=False)
        self.checkpoint_sha256 = sha256(checkpoint)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self._lock = threading.Lock()
        self.candidate=None
        self.candidate_threshold=float(candidate_threshold)
        self.candidate_weight=None if candidate_weight is None else float(candidate_weight)
        if self.candidate_weight is not None and (candidate_checkpoint is None or not 0<self.candidate_weight<1):
            raise ValueError('La combinación requiere un segundo checkpoint y un peso entre cero y uno.')
        self.candidate_checkpoint_sha256=None
        if candidate_checkpoint is not None:
            if self.candidate_weight is None and not 0<self.candidate_threshold<=1:
                raise ValueError('Confianza candidata inválida.')
            self.candidate=InferenceService(candidate_checkpoint,cpu_threads)
            if self.candidate.classes!=self.classes:raise ValueError('Clases candidatas incompatibles.')
            self.candidate_checkpoint_sha256=self.candidate.checkpoint_sha256
            self.parameter_count+=self.candidate.parameter_count
        self.refiner = None
        self.refiner_checkpoint_sha256 = None
        if refiner_checkpoint is not None:
            if refiner_weight is None or not 0 < float(refiner_weight) <= 1:
                raise ValueError('El refinador requiere un peso mayor que cero y hasta uno.')
            self.refiner = InferenceService(refiner_checkpoint, cpu_threads)
            if self.refiner.classes != self.classes:
                raise ValueError('Clases incompatibles en el refinador.')
            self.refiner_weight = float(refiner_weight)
            self.refiner_preserve = frozenset(refiner_preserve)
            self.refiner_preserve_consensus = frozenset(refiner_preserve_consensus)
            if not (self.refiner_preserve | self.refiner_preserve_consensus) <= set(self.classes):
                raise ValueError('Materiales inválidos en el refinador.')
            self.refiner_checkpoint_sha256 = self.refiner.checkpoint_sha256
            self.parameter_count += self.refiner.parameter_count
            self.refiner_calibration = None
            if refiner_calibration:
                calibration = json.loads(Path(refiner_calibration).read_text(encoding='utf-8'))
                expected = {'refined': self.checkpoint_sha256, 'scene': self.candidate_checkpoint_sha256,
                            'transformer': self.refiner_checkpoint_sha256}
                if (calibration.get('classes') != list(self.classes) or calibration.get('checkpoints') != expected or
                        calibration.get('scene_weight') != self.candidate_weight or calibration.get('test_used') is not False):
                    raise ValueError('La calibración no corresponde a los checkpoints activos.')
                matrix = np.asarray(calibration['matrix'], dtype=np.float64)
                bias = np.asarray(calibration['bias'], dtype=np.float64)
                count = len(self.classes)
                if (matrix.shape != (2 * count, count) or bias.shape != (count,) or
                        not np.isfinite(matrix).all() or not np.isfinite(bias).all()):
                    raise ValueError('La calibración tiene coeficientes inválidos.')
                self.refiner_calibration = matrix, bias

    def _refine(self, rgb, reading, started, agreement=None):
        if (self.refiner is None or reading.material in self.refiner_preserve or
                (agreement == reading.material and agreement in self.refiner_preserve_consensus)):
            return reading
        refinement = self.refiner.classify(rgb, region=reading.region)
        proposed = refinement.probabilities
        if self.refiner_calibration is not None:
            matrix, bias = self.refiner_calibration
            inputs = [p[label] for p in (reading.probabilities, proposed) for label in self.classes]
            logits = np.log(np.clip(inputs, 1e-6, 1)) @ matrix + bias
            values = np.exp(logits - logits.max())
            proposed = dict(zip(self.classes, (values / values.sum()).tolist()))
        weight = self.refiner_weight
        probabilities = {label: (1 - weight) * reading.probabilities[label] + weight * proposed[label]
                         for label in self.classes}
        material = max(probabilities, key=probabilities.get)
        return replace(reading, material=material, confidence=probabilities[material], probabilities=probabilities,
                       inference_ms=(time.perf_counter() - started) * 1000, views=reading.views + refinement.views)

    def preprocess(self, rgb, region):
        return self.transform(Image.fromarray(crop_region(rgb, region))).unsqueeze(0)

    def classify(self, rgb, fraction=.72, use_contours=False, *, region=None, mode="full", manual_bounds=(.2, .2, .8, .8)):
        if torch.get_num_threads()!=self.cpu_threads:torch.set_num_threads(self.cpu_threads)
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 16:
            raise ValueError("Se requiere una imagen RGB válida.")
        if not .4 <= fraction <= .95:
            raise ValueError("Tamaño de región inválido.")
        started = time.perf_counter()
        candidate_reading=None
        if self.candidate is not None:
            candidate_reading=self.candidate.classify(rgb,fraction,use_contours,region=region,mode=mode,manual_bounds=manual_bounds)
            if self.candidate_weight is None and candidate_reading.confidence>=self.candidate_threshold:
                return self._refine(rgb, candidate_reading, started)
            region=candidate_reading.region
        if region is None:
            result = Localizer("auto" if use_contours else mode, manual_bounds).locate(rgb)
            if result.region is None:
                raise LocalizationError(result.message)
            region = result.region
        crop = crop_region(rgb, region)
        tensor = self.preprocess(rgb, region)
        with self._lock, torch.inference_mode():
            probabilities = model_probabilities(self.model, tensor, self.metadata["config"])[0].tolist()
        agreement = None
        if candidate_reading is not None and self.candidate_weight is not None:
            weight=self.candidate_weight
            if self.refiner is not None and self.refiner_preserve_consensus:
                base = self.classes[max(range(len(probabilities)), key=probabilities.__getitem__)]
                if base == candidate_reading.material:
                    agreement = base
            probabilities=[(1-weight)*probability+weight*candidate_reading.probabilities[label]
                           for label,probability in zip(self.classes,probabilities)]
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        reading = Reading(self.classes[index], probabilities[index], dict(zip(self.classes, probabilities)), region,
                          (time.perf_counter() - started) * 1000, hashlib.sha256(crop.tobytes()).hexdigest(),
                          hashlib.sha256(tensor.numpy().tobytes()).hexdigest(), self.metadata["image_size"],
                          (2 if self.metadata["config"].get("inference_hflip",False) else 1)+
                          (candidate_reading.views if candidate_reading else 0))
        return self._refine(rgb, reading, started, agreement)
