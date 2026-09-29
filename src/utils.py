"""
Shared utilities: config loading, validation, logging, device selection.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

LOGGER_NAME = "YOLO_QAT"

# section -> required keys
_REQUIRED: Dict[str, tuple] = {
    "model": ("weights", "input_size"),
    "dataset": ("data_yaml",),
    "calibration": ("num_batches", "batch_size"),
    "qat_training": ("epochs", "batch_size", "learning_rate"),
    "export": ("onnx_path",),
    "output": ("run_dir",),
}


def setup_logging(log_dir: str = "runs", level: int = logging.INFO) -> logging.Logger:
    """Configure the project logger for console and file output."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Reconfigure cleanly if called twice (e.g. pipeline then a sub-command).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    log_file = logging.FileHandler(os.path.join(log_dir, "qat_pipeline.log"))
    log_file.setLevel(logging.DEBUG)
    log_file.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    logger.addHandler(log_file)

    logger.propagate = False
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} did not parse to a mapping.")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate structure and paths up front.

    QAT runs are long. Every check here is something that would otherwise
    surface after calibration, or worse, after training.
    """
    missing_sections = [s for s in _REQUIRED if s not in config]
    if missing_sections:
        raise ValueError(f"Config is missing section(s): {', '.join(missing_sections)}")

    for section, keys in _REQUIRED.items():
        absent = [k for k in keys if k not in config[section]]
        if absent:
            raise ValueError(f"Config section '{section}' is missing key(s): {', '.join(absent)}")

    weights = config["model"]["weights"]
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Model weights not found: {weights}. Place yolov8s.pt in weights/ or "
            "update model.weights."
        )

    data_yaml = config["dataset"]["data_yaml"]
    if not os.path.exists(data_yaml) and not _is_bundled_dataset(data_yaml):
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Point dataset.data_yaml at your "
            "YOLO data.yaml (it needs train:, val:, nc: and names:), or use one of "
            "ultralytics' bundled dataset names such as 'coco8.yaml'."
        )

    input_size = config["model"]["input_size"]
    if not (isinstance(input_size, (list, tuple)) and len(input_size) == 2):
        raise ValueError(f"model.input_size must be [height, width], got {input_size!r}")
    if any(int(dim) % 32 for dim in input_size):
        raise ValueError(
            f"model.input_size {list(input_size)} must be a multiple of 32 "
            "(YOLOv8's maximum stride)."
        )

    lr = float(config["qat_training"]["learning_rate"])
    if lr > 1e-3:
        logging.getLogger(LOGGER_NAME).warning(
            "qat_training.learning_rate=%.1e is high for QAT. This is a fine-tune of "
            "converged weights; 1e-5 to 1e-4 is the usual range, and a large LR "
            "undoes the pretrained weights faster than quantization noise is learned.",
            lr,
        )


def _is_bundled_dataset(name: str) -> bool:
    """
    True for a dataset ultralytics ships by name (``coco8.yaml``, ``VOC.yaml``, ...).

    These are not local paths; ultralytics resolves and downloads them itself,
    so a plain ``os.path.exists`` check would reject a perfectly valid config.
    """
    if os.sep in name or (os.altsep and os.altsep in name):
        return False
    try:
        from ultralytics.cfg import ROOT
    except ImportError:  # pragma: no cover
        return False
    return (Path(ROOT) / "cfg" / "datasets" / name).exists()


def get_device(device_str: str) -> torch.device:
    """
    Resolve a device string, accepting ultralytics' conventions.

    Accepts "0", "cuda:0", "cpu", "mps" and "" (auto).
    """
    text = str(device_str).strip().lower()

    if text in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if text == "cpu":
        return torch.device("cpu")

    if text == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")

    device = torch.device(f"cuda:{text}" if text.isdigit() else text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_str!r} requested but CUDA is unavailable. "
            "Install a CUDA build of PyTorch, or set qat_training.device to 'cpu'."
        )
    return device


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def ensure_dir(path: str) -> str:
    """Create a directory if needed and return it."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def section(logger: logging.Logger, title: str, *details: str) -> None:
    """Log a visually distinct pipeline step header."""
    logger.info("")
    logger.info("─" * 68)
    logger.info(title)
    for detail in details:
        logger.info("  %s", detail)
    logger.info("─" * 68)
