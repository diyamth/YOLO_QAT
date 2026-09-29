"""
Accuracy measurement — the part that tells you whether QAT actually worked.

The original pipeline validated on *loss* only, which cannot answer the one
question that matters: did quantization cost accuracy, and did fine-tuning win
it back? Loss is not comparable across quantization states, and a detector's
quality lives in mAP.

Three tools here:

* :func:`evaluate_map` - mAP for a model in its current quantization state.
* :func:`compare_precision` - FP32 vs INT8 mAP on the *same weights*, which is
  the honest measure of what quantization cost.
* :func:`layer_sensitivity` - per-layer mAP recovery, to find the few layers
  that should stay FP32.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionValidator

from .qat_quantize import (
    count_quantizers,
    is_quantized,
    iter_quantizers,
    quantization_disabled,
)

logger = logging.getLogger("YOLO_QAT")

# Ultralytics has used a few spellings for the primary metric across releases.
_MAP_KEYS = ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95(B)", "mAP50-95")
_MAP50_KEYS = ("metrics/mAP50(B)", "metrics/mAP50", "mAP50(B)", "mAP50")


def _pick(metrics: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    """Return the first present metric, tolerating ultralytics key renames."""
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def evaluate_map(
    model: nn.Module,
    data_yaml: str,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    split: str = "val",
    workers: int = 8,
    plots: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run detection validation and return the metrics dict.

    Evaluates the model exactly as it currently stands: if quantizers are
    enabled, this is simulated-INT8 mAP; if disabled, FP32 mAP.

    Args:
        model: Model to evaluate.
        data_yaml: Path to the YOLO ``data.yaml``.
        imgsz: Inference resolution.
        batch: Validation batch size.
        device: Device string.
        split: Dataset split to evaluate.
        workers: DataLoader workers.
        plots: Write validation plots.
        verbose: Per-class metric output.

    Returns:
        Metrics dict, plus normalised ``mAP50-95`` / ``mAP50`` keys.
    """
    args = {
        "data": data_yaml,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "split": split,
        "workers": workers,
        "plots": plots,
        "verbose": verbose,
        "save_json": False,
        "save_txt": False,
        # fp16 would quantize on top of the simulated INT8 grid and muddy the
        # comparison; keep validation in fp32 throughout.
        "half": False,
    }

    validator = DetectionValidator(args=args)
    metrics = dict(validator(model=model))

    metrics["mAP50-95"] = _pick(metrics, _MAP_KEYS)
    metrics["mAP50"] = _pick(metrics, _MAP50_KEYS)
    return metrics


def compare_precision(
    model: nn.Module,
    data_yaml: str,
    **eval_kwargs: Any,
) -> Dict[str, Any]:
    """
    Measure FP32 and simulated-INT8 mAP on the same weights.

    Running both on one model isolates the cost of quantization from every other
    variable - same weights, same data, same preprocessing. That difference is
    what QAT is trying to close.

    Returns:
        ``{"fp32": {...}, "int8": {...}, "delta_map": float, "retention_pct": float}``
    """
    if not is_quantized(model):
        raise RuntimeError("Model carries no quantizers; nothing to compare.")

    logger.info("Evaluating FP32 baseline (quantizers disabled)...")
    with quantization_disabled(model):
        fp32 = evaluate_map(model, data_yaml, **eval_kwargs)

    logger.info("Evaluating simulated INT8 (quantizers enabled)...")
    int8 = evaluate_map(model, data_yaml, **eval_kwargs)

    fp32_map = fp32.get("mAP50-95") or 0.0
    int8_map = int8.get("mAP50-95") or 0.0
    delta = int8_map - fp32_map
    retention = (int8_map / fp32_map * 100.0) if fp32_map else float("nan")

    logger.info("=" * 62)
    logger.info("  FP32 mAP50-95 : %.4f", fp32_map)
    logger.info("  INT8 mAP50-95 : %.4f", int8_map)
    logger.info("  Delta         : %+.4f (%.2f%% retained)", delta, retention)
    logger.info("=" * 62)

    return {
        "fp32": fp32,
        "int8": int8,
        "fp32_map": fp32_map,
        "int8_map": int8_map,
        "delta_map": delta,
        "retention_pct": retention,
    }


def _layer_groups(model: nn.Module) -> Dict[str, List[Any]]:
    """
    Group quantizers by the layer they belong to.

    ``model.0.conv.input_quantizer`` and ``model.0.conv.weight_quantizer`` both
    belong to layer ``model.0.conv``; sensitivity is a property of the layer, so
    they are toggled together.
    """
    groups: Dict[str, List[Any]] = defaultdict(list)
    for name, quantizer in iter_quantizers(model):
        layer = name.rsplit(".", 1)[0] if "." in name else name
        groups[layer].append(quantizer)
    return dict(groups)


@torch.no_grad()
def layer_sensitivity(
    model: nn.Module,
    data_yaml: str,
    top_k: int = 10,
    **eval_kwargs: Any,
) -> List[Tuple[str, float]]:
    """
    Rank layers by how much mAP is recovered by leaving them in FP32.

    Quantizes everything, then disables one layer at a time and re-measures.
    Layers whose exclusion recovers the most mAP are the quantization-hostile
    ones — feed them back as ``quantization.skip_patterns`` and retrain.

    This costs one full validation pass **per layer**, so point it at a small
    split or a subset. It is a diagnostic, not part of the default pipeline.

    Args:
        model: Calibrated, quantized model.
        data_yaml: Path to the YOLO ``data.yaml``.
        top_k: How many of the worst layers to report.
        **eval_kwargs: Forwarded to :func:`evaluate_map`.

    Returns:
        ``[(layer_name, map_gain), ...]`` sorted by gain, highest first.
    """
    groups = _layer_groups(model)
    logger.info(
        "Sensitivity analysis over %d layer(s) - this runs %d validation passes",
        len(groups),
        len(groups) + 1,
    )

    baseline = evaluate_map(model, data_yaml, **eval_kwargs).get("mAP50-95") or 0.0
    logger.info("Fully-quantized baseline mAP50-95: %.4f", baseline)

    results: List[Tuple[str, float]] = []
    for index, (layer, quantizers) in enumerate(groups.items(), start=1):
        previous = [(q, q.is_enabled) for q in quantizers]
        for q in quantizers:
            q.disable()
        try:
            score = evaluate_map(model, data_yaml, **eval_kwargs).get("mAP50-95") or 0.0
        finally:
            for q, was_enabled in previous:
                if was_enabled:
                    q.enable()

        gain = score - baseline
        results.append((layer, gain))
        logger.info("  [%d/%d] %-44s %+.4f", index, len(groups), layer, gain)

    results.sort(key=lambda item: item[1], reverse=True)

    logger.info("Top %d quantization-sensitive layers:", min(top_k, len(results)))
    for layer, gain in results[:top_k]:
        logger.info("  %-44s recovers %+.4f mAP", layer, gain)

    return results


def log_quantization_state(model: nn.Module, context: str = "") -> None:
    """Log a one-line quantizer census, for the training log."""
    stats = count_quantizers(model)
    logger.info(
        "Quantizer state%s: %d total, %d enabled, %d calibrated",
        f" ({context})" if context else "",
        stats["total"],
        stats["enabled"],
        stats["calibrated"],
    )
