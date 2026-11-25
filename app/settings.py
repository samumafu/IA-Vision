"""Configuracion portable de la aplicacion, sin modificar checkpoints."""
import json
import math
import os
from pathlib import Path

from utils.config import PROJECT_ROOT


def load_settings():
    settings = json.loads((PROJECT_ROOT / "config/app.json").read_text(encoding="utf-8"))
    override = os.environ.get("VISION_IA_CHECKPOINT")
    if override and (PROJECT_ROOT / Path(override).expanduser()).resolve() != (PROJECT_ROOT / settings['checkpoint']).resolve():
        settings['checkpoint'] = override
        if settings.get('material_refiner_calibration'):
            settings.pop('material_refiner_checkpoint', None)
    return settings


def checkpoint_signature(value):
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("No se encontró el checkpoint. Revisa la ruta en Opciones o VISION_IA_CHECKPOINT.")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def validate_options(fraction, interval):
    if fraction is not None and (not math.isfinite(fraction) or not .4 <= fraction <= .95):
        raise ValueError("La región central debe ocupar entre 40 % y 95 % del lado menor.")
    if not math.isfinite(interval) or not .1 <= interval <= 5:
        raise ValueError("El intervalo de lectura debe estar entre 0,1 y 5 segundos.")
