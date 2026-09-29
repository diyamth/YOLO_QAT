"""
QAT Quantization — insert fake-quantization (Q/DQ) nodes into YOLOv8.

Backend: **NVIDIA TensorRT Model Optimizer** (``nvidia-modelopt``).

Why not ``pytorch-quantization``
--------------------------------
The original implementation used ``pytorch_quantization.quant_modules.initialize()``,
which rebinds names in the ``torch.nn`` namespace (``torch.nn.Conv2d -> QuantConv2d``).
That only affects layers *constructed* afterwards by attribute lookup. A YOLOv8
``.pt`` is a **pickled model object**: unpickling resolves
``torch.nn.modules.conv.Conv2d`` directly and never calls ``__init__``, so the
monkey-patch does nothing. Measured on ``yolov8s.pt``: 64 Conv2d layers, 0 patched.
The pipeline could never have produced a quantized model.

``pytorch-quantization`` is also in maintenance mode, ships Linux/x86_64 CUDA
wheels only, and collides with ultralytics >= 8.4 — whose checkpoint writer calls
``qat_state()``, which detects any class named ``TensorQuantizer`` and then hands
it to ModelOpt, crashing on every save.

ModelOpt is the maintained successor, is what ultralytics >= 8.4 integrates with
natively, and inserts quantizers by explicit module surgery, so it works on a
pickled model.

What gets quantized
-------------------
``INT8_DEFAULT_CFG`` gives every Conv/Linear an **input quantizer** (per-tensor)
and a **weight quantizer** (per-output-channel, ``axis=0``), and leaves *output*
quantizers disabled. That is deliberate and it is what TensorRT wants: quantizing
inputs and weights lets TRT fuse Q/DQ into the convolution itself, while an
output quantizer would pin a scale mid-graph and block that fusion.

The DFL conv in the detect head is excluded by default. It holds fixed ``arange``
weights implementing a softmax-weighted sum; INT8 there costs accuracy and saves
nothing. Ultralytics itself always freezes ``.dfl``.

Verified on yolov8s: 308 quantizers inserted, exporting to 131 QuantizeLinear +
131 DequantizeLinear ONNX nodes.
"""

from __future__ import annotations

import copy
import fnmatch
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger("YOLO_QAT")

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer

    MODELOPT_AVAILABLE = True
    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - depends on host
    MODELOPT_AVAILABLE = False
    _IMPORT_ERROR = exc
    mtq = None  # type: ignore[assignment]
    TensorQuantizer = None  # type: ignore[assignment]


_INSTALL_HINT = (
    "nvidia-modelopt is required for QAT but is not installed.\n"
    "  pip install nvidia-modelopt\n"
    "See https://github.com/NVIDIA/TensorRT-Model-Optimizer"
)

# Layers left in FP32. fnmatch patterns matched against the quantizer name.
DEFAULT_SKIP_PATTERNS: Tuple[str, ...] = ("*dfl*",)

# Calibration algorithms ModelOpt accepts for INT8 activation ranges.
_VALID_ALGORITHMS = {"max", "entropy", "percentile", "mse"}


def require_quantization_backend() -> None:
    """Raise a helpful error if ModelOpt is missing."""
    if not MODELOPT_AVAILABLE:
        raise ImportError(f"{_INSTALL_HINT}\n\nOriginal import error: {_IMPORT_ERROR}")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════


def build_quant_config(
    algorithm: str = "max",
    percentile: float = 99.99,
    skip_patterns: Sequence[str] = DEFAULT_SKIP_PATTERNS,
) -> Dict[str, Any]:
    """
    Build the ModelOpt INT8 quantization config for YOLOv8.

    Starts from ``mtq.INT8_DEFAULT_CFG`` (per-channel weights, per-tensor
    activations, output quantizers off) and appends the layer exclusions.

    Args:
        algorithm: Activation calibration method - "max", "entropy",
            "percentile" or "mse".

            * ``max``   - fastest, tracks the absolute maximum. The NVIDIA
              default and a solid baseline.
            * ``percentile`` - clips at ``percentile``, which helps when a few
              outlier activations would otherwise stretch the range and coarsen
              every step.
            * ``entropy`` / ``mse`` - minimise KL divergence / squared error
              against the FP32 distribution. Slower to calibrate.
        percentile: Clip percentile, used only when ``algorithm="percentile"``.
        skip_patterns: Quantizer-name patterns to leave in FP32.

    Returns:
        A ModelOpt config dict ready for ``mtq.quantize``.
    """
    require_quantization_backend()

    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown calibration algorithm {algorithm!r}. Valid: {sorted(_VALID_ALGORITHMS)}"
        )

    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    # ModelOpt changed quant_cfg from a dict to a list of entries; support both.
    quant_cfg = config["quant_cfg"]
    for pattern in skip_patterns:
        if isinstance(quant_cfg, list):
            quant_cfg.append({"quantizer_name": pattern, "enable": False})
        elif isinstance(quant_cfg, dict):
            quant_cfg[pattern] = {"enable": False}
        else:  # pragma: no cover - guards a future format change
            raise TypeError(f"Unexpected quant_cfg type {type(quant_cfg).__name__}")

    config["algorithm"] = (
        {"method": "percentile", "percentile": percentile}
        if algorithm == "percentile"
        else algorithm
    )

    logger.info(
        "Quant config: INT8, weights=per-channel, activations=per-tensor, "
        "calibration=%s%s, skipping %s",
        algorithm,
        f" (p={percentile})" if algorithm == "percentile" else "",
        ", ".join(skip_patterns) if skip_patterns else "nothing",
    )
    return config


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════


def _extract_images(batch: Any) -> torch.Tensor:
    """
    Pull the image tensor out of an ultralytics batch and normalise it.

    Ultralytics dataloaders yield ``uint8`` images in a dict; the model expects
    float in [0, 1]. Bare tensors are passed through for custom loaders.
    """
    images = batch["img"] if isinstance(batch, dict) else batch
    if not torch.is_tensor(images):
        raise TypeError(f"Expected a tensor or a dict with 'img', got {type(batch).__name__}")
    return images.float() / 255.0 if images.dtype == torch.uint8 else images.float()


def make_calibration_loop(
    dataloader: Iterable[Any],
    device: torch.device,
    num_batches: int,
) -> Callable[[nn.Module], None]:
    """
    Build the forward loop ModelOpt runs to observe activation ranges.

    ModelOpt calls this with quantizers in *collect* mode, then computes each
    range from what it saw. This is **not** PTQ - it only seeds the ranges.
    QAT fine-tuning afterwards is what actually recovers accuracy.

    Args:
        dataloader: Yields ultralytics batches or raw tensors.
        device: Device to run calibration on.
        num_batches: How many batches to observe.
    """

    def forward_loop(model: nn.Module) -> None:
        seen = 0
        progress = tqdm(total=num_batches, desc="Calibrating", unit="batch")
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if seen >= num_batches:
                        break
                    model(_extract_images(batch).to(device, non_blocking=True))
                    seen += 1
                    progress.update(1)
        finally:
            progress.close()

        if seen == 0:
            raise RuntimeError(
                "Calibration dataloader yielded no batches - check dataset.data_yaml."
            )
        if seen < num_batches:
            logger.warning(
                "Dataset exhausted after %d/%d calibration batches; ranges are "
                "estimated from less data than requested.",
                seen,
                num_batches,
            )

    return forward_loop


def quantize_model(
    model: nn.Module,
    quant_config: Dict[str, Any],
    calibration_loop: Callable[[nn.Module], None],
) -> nn.Module:
    """
    Insert Q/DQ nodes and calibrate their ranges, in one pass.

    Args:
        model: The model to convert (modified in place and returned).
        quant_config: Config from :func:`build_quant_config`.
        calibration_loop: Forward loop from :func:`make_calibration_loop`.

    Returns:
        The quantization-aware model.
    """
    require_quantization_backend()

    was_training = model.training
    model.eval()  # calibrate on inference-mode statistics (BN running stats)
    try:
        model = mtq.quantize(model, quant_config, forward_loop=calibration_loop)
    finally:
        if was_training:
            model.train()

    stats = count_quantizers(model)
    logger.info(
        "Quantized: %d quantizers (%d enabled, %d calibrated)",
        stats["total"],
        stats["enabled"],
        stats["calibrated"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Quantizer introspection and control
# ══════════════════════════════════════════════════════════════════════════════


def iter_quantizers(model: nn.Module) -> Iterable[Tuple[str, Any]]:
    """Yield ``(name, TensorQuantizer)`` for every quantizer in the model."""
    if not MODELOPT_AVAILABLE:
        return
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            yield name, module


def count_quantizers(model: nn.Module) -> Dict[str, int]:
    """Count total / enabled / calibrated quantizers."""
    total = enabled = calibrated = 0
    for _, q in iter_quantizers(model):
        total += 1
        if q.is_enabled:
            enabled += 1
            if getattr(q, "amax", None) is not None:
                calibrated += 1
    return {"total": total, "enabled": enabled, "calibrated": calibrated}


def is_quantized(model: nn.Module) -> bool:
    """True if the model carries any fake-quantization."""
    return any(True for _ in iter_quantizers(model))


def disable_quantizers_matching(model: nn.Module, patterns: Sequence[str]) -> List[str]:
    """
    Disable quantizers whose name matches any pattern.

    Used to roll back layers that sensitivity analysis flags as
    quantization-hostile, without rebuilding the model.
    """
    disabled = []
    for name, q in iter_quantizers(model):
        if any(fnmatch.fnmatch(name, p) for p in patterns):
            q.disable()
            disabled.append(name)
    if disabled:
        logger.info("Disabled %d quantizer(s) matching %s", len(disabled), list(patterns))
    return disabled


def set_quantizer_state(model: nn.Module, enabled: bool) -> int:
    """
    Enable or disable *all* quantizers.

    Disabling turns the model back into plain FP32, which is how the baseline
    mAP is measured on the exact same weights.
    """
    count = 0
    for _, q in iter_quantizers(model):
        q.enable() if enabled else q.disable()
        count += 1
    return count


class quantization_disabled:
    """
    Context manager that temporarily runs the model in FP32.

    Restores each quantizer's previous state on exit, so a model with
    sensitivity-driven per-layer exclusions is not flattened by a baseline run.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._previous: List[Tuple[Any, bool]] = []

    def __enter__(self) -> nn.Module:
        for _, q in iter_quantizers(self.model):
            self._previous.append((q, q.is_enabled))
            q.disable()
        return self.model

    def __exit__(self, *exc_info: Any) -> None:
        for q, was_enabled in self._previous:
            if was_enabled:
                q.enable()
        self._previous.clear()


def assert_qat_active(model: nn.Module, context: str = "") -> Dict[str, int]:
    """
    Fail loudly if the model is not actually quantization-aware.

    Guards against silently training or exporting an FP32 model and calling it
    QAT - the exact failure this project exists to avoid, and the one the
    original implementation shipped with.
    """
    require_quantization_backend()
    stats = count_quantizers(model)
    suffix = f" ({context})" if context else ""

    if stats["total"] == 0:
        raise RuntimeError(
            f"No quantizers found{suffix}. The model was never converted - "
            "call quantize_model() before training or exporting."
        )
    if stats["enabled"] == 0:
        raise RuntimeError(
            f"All {stats['total']} quantizers are disabled{suffix}. "
            "That would be plain FP32 training, not QAT."
        )
    if stats["calibrated"] == 0:
        raise RuntimeError(
            f"No quantizer has a calibrated range{suffix}. Calibration did not run "
            "or saw no data; training from uninitialised ranges diverges immediately."
        )
    return stats


def quantization_summary(model: nn.Module, max_rows: int = 12) -> str:
    """Render a short human-readable table of the quantizer state."""
    rows = []
    for name, q in iter_quantizers(model):
        if not q.is_enabled:
            continue
        amax = getattr(q, "amax", None)
        if amax is None:
            rng = "uncalibrated"
        elif amax.numel() == 1:
            rng = f"±{amax.item():.4f}"
        else:
            rng = f"±[{amax.min().item():.4f}, {amax.max().item():.4f}] ({amax.numel()} ch)"
        rows.append((name, f"{q.num_bits}-bit", rng))

    stats = count_quantizers(model)
    lines = [
        f"Quantizers: {stats['total']} total, {stats['enabled']} enabled, "
        f"{stats['calibrated']} calibrated"
    ]
    width = max((len(r[0]) for r in rows[:max_rows]), default=0)
    for name, bits, rng in rows[:max_rows]:
        lines.append(f"  {name:<{width}}  {bits}  {rng}")
    if len(rows) > max_rows:
        lines.append(f"  ... and {len(rows) - max_rows} more")
    return "\n".join(lines)
