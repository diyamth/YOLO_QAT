"""
Dataset plumbing shared by calibration and evaluation.

The important decision here is that **calibration uses the validation
transform pipeline applied to training images**.

Calibration measures the range of activations the model will see at inference
time. Feeding it mosaic/mixup/HSV-augmented composites - as ``mode="train"``
does - measures the range of imagery that only ever exists during training:
four-image collages with different statistics from any real frame. The ranges
come out too wide, every quantization step gets coarser, and accuracy drops for
no reason. So: training *images*, validation *transforms*.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional, Tuple

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG

logger = logging.getLogger("YOLO_QAT")


def load_dataset_dict(data_yaml: str) -> Dict[str, Any]:
    """Resolve a YOLO ``data.yaml`` into ultralytics' dataset dict."""
    return check_det_dataset(data_yaml)


def _call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Call ``fn`` passing only the keyword arguments it actually accepts.

    Ultralytics changes these signatures between minor releases (``device`` and
    ``pin_memory`` on ``build_dataloader`` are recent additions). Filtering
    keeps this working across versions instead of hard-failing on an unexpected
    keyword.
    """
    try:
        supported = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        logger.debug("Dropped unsupported kwargs for %s: %s", fn.__name__, sorted(dropped))
    return fn(*args, **filtered)


def build_eval_dataloader(
    data_yaml: str,
    split: str = "train",
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 8,
    device: str = "cpu",
    shuffle: bool = False,
    rect: bool = False,
    fraction: Optional[float] = None,
    data_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a dataloader with **validation-style preprocessing** (no augmentation).

    Args:
        data_yaml: Path to the YOLO ``data.yaml``.
        split: Which split's images to read - "train", "val" or "test".
        imgsz: Inference resolution.
        batch_size: Batch size.
        workers: DataLoader worker processes.
        device: Device string, forwarded to ultralytics for pinning decisions.
        shuffle: Shuffle the data. Worth enabling for calibration so a capped
            batch count samples across the dataset rather than reading the
            first N files, which are often correlated (same scene, same class).
        rect: Rectangular batching. Leave off for calibration so every image is
            letterboxed to the exact deployment resolution.
        fraction: Optionally use only a fraction of the split.
        data_dict: Pre-resolved dataset dict, to avoid re-reading the yaml.

    Returns:
        ``(dataloader, data_dict)``
    """
    data = data_dict if data_dict is not None else load_dataset_dict(data_yaml)

    if split not in data or not data[split]:
        available = [k for k in ("train", "val", "test") if data.get(k)]
        raise KeyError(
            f"Split {split!r} is not defined in {data_yaml}. Available splits: {available}"
        )

    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml
    cfg.imgsz = imgsz
    cfg.batch = batch_size
    cfg.workers = workers
    cfg.rect = rect
    cfg.cache = False
    cfg.fraction = fraction if fraction is not None else 1.0

    # mode="val" is what disables mosaic/mixup/copy-paste/HSV/flips. Zero the
    # augmentation hyperparameters too, so this still holds if a future
    # ultralytics version reads them in val mode.
    for key, value in (
        ("mosaic", 0.0),
        ("mixup", 0.0),
        ("cutmix", 0.0),
        ("copy_paste", 0.0),
        ("erasing", 0.0),
        ("hsv_h", 0.0),
        ("hsv_s", 0.0),
        ("hsv_v", 0.0),
        ("degrees", 0.0),
        ("translate", 0.0),
        ("scale", 0.0),
        ("shear", 0.0),
        ("perspective", 0.0),
        ("flipud", 0.0),
        ("fliplr", 0.0),
        ("auto_augment", None),
    ):
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    dataset = _call_with_supported_kwargs(
        build_yolo_dataset,
        cfg,
        data[split],
        batch_size,
        data,
        mode="val",
        rect=rect,
        stride=32,
    )

    dataloader = _call_with_supported_kwargs(
        build_dataloader,
        dataset,
        batch_size,
        workers,
        shuffle=shuffle,
        rank=-1,
        device=device,
    )

    logger.info(
        "Built %s dataloader: %d images, batch=%d, imgsz=%d, augmentation=off",
        split,
        len(dataset),
        batch_size,
        imgsz,
    )
    return dataloader, data
