"""
QAT pipeline orchestrator.

    FP32 weights
        │
        ├── [1] baseline mAP ........... what we are trying not to lose
        │
        ├── [2] insert Q/DQ + calibrate  ranges seeded from real data
        │
        ├── [3] post-calibration mAP ... this is the PTQ number
        │
        ├── [4] QAT fine-tune .......... weights adapt to quantization noise
        │
        ├── [5] final INT8 mAP ......... QAT vs PTQ is the payoff
        │
        ├── [6] export ONNX with Q/DQ
        │
        └── [7] build TensorRT INT8 engine

Steps 1, 3 and 5 exist because "is this QAT or PTQ?" is an empirical question.
Step 3 *is* post-training quantization: calibrated ranges, no retraining. If
step 5 does not beat step 3, the fine-tuning contributed nothing and something
is wrong. The original pipeline measured none of this.

Usage:
    python -m src.run_qat_pipeline --config configs/qat_config.yaml
    python -m src.run_qat_pipeline --config configs/qat_config.yaml --skip-engine
    python -m src.run_qat_pipeline --config configs/qat_config.yaml --sensitivity
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

if __package__ in (None, ""):  # allow `python src/run_qat_pipeline.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from .build_trt_engine import build_engine
from .qat_data import build_eval_dataloader
from .qat_export import export_qat_onnx
from .qat_quantize import (
    DEFAULT_SKIP_PATTERNS,
    build_quant_config,
    make_calibration_loop,
    quantization_summary,
    quantize_model,
    require_quantization_backend,
)
from .qat_trainer import QATTrainer, build_training_overrides
from .qat_validate import evaluate_map, layer_sensitivity
from .utils import (
    ensure_dir,
    get_device,
    load_config,
    section,
    set_seed,
    setup_logging,
    validate_config,
)

logger = logging.getLogger("YOLO_QAT")


def _evaluation_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Shared arguments for every mAP measurement, so they stay comparable."""
    return {
        "imgsz": config["model"]["input_size"][0],
        "batch": config["qat_training"].get("val_batch_size", config["qat_training"]["batch_size"]),
        "device": config["qat_training"].get("device", "0"),
        "workers": config["qat_training"].get("workers", 8),
        "split": config["dataset"].get("val_split", "val"),
    }


def _load_trained_model(trainer: QATTrainer) -> torch.nn.Module:
    """
    Recover the best QAT weights after training.

    Ultralytics writes checkpoints as fp16 and moves the ModelOpt quantization
    out of the pickle into a sidecar ``modelopt`` entry (runtime-generated
    classes cannot be pickled). Loading through the ultralytics API reverses
    both, so prefer ``best.pt``; fall back to the in-memory EMA if it is
    unreadable for any reason.
    """
    from ultralytics import YOLO

    from .qat_quantize import count_quantizers

    best = getattr(trainer, "best", None)
    if best is not None and os.path.exists(best):
        try:
            model = YOLO(str(best)).model.float()
            stats = count_quantizers(model)
            if stats["enabled"] > 0:
                logger.info(
                    "Loaded best checkpoint %s (%d quantizers restored)", best, stats["total"]
                )
                return model
            logger.warning(
                "best.pt loaded without active quantizers; falling back to the in-memory model."
            )
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the run
            logger.warning("Could not load %s (%s); using the in-memory model.", best, exc)

    ema = getattr(getattr(trainer, "ema", None), "ema", None)
    model = ema if ema is not None else trainer.model
    return model.float()


def run_pipeline(
    config_path: str,
    skip_engine: bool = False,
    skip_baseline: bool = False,
    run_sensitivity: bool = False,
) -> Dict[str, Any]:
    """Run the full QAT pipeline and return a summary dict."""
    config = load_config(config_path)
    run_dir = ensure_dir(config["output"]["run_dir"])
    setup_logging(run_dir)

    logger.info("=" * 68)
    logger.info("YOLOv8 Quantization-Aware Training pipeline")
    logger.info("=" * 68)

    validate_config(config)
    require_quantization_backend()

    set_seed(config.get("seed", 0))
    device = get_device(config["qat_training"].get("device", "0"))
    logger.info("Device: %s", device)

    started = time.time()
    results: Dict[str, Any] = {}
    eval_kwargs = _evaluation_kwargs(config)
    data_yaml = config["dataset"]["data_yaml"]
    imgsz = config["model"]["input_size"][0]

    # ── 1. FP32 baseline ────────────────────────────────────────────────────
    from ultralytics import YOLO

    section(logger, "STEP 1/7  Load model and measure the FP32 baseline")
    yolo = YOLO(config["model"]["weights"])
    model = yolo.model.float().to(device)

    if skip_baseline:
        logger.info("Baseline skipped (--skip-baseline)")
        results["fp32_map"] = None
    else:
        baseline = evaluate_map(model, data_yaml, **eval_kwargs)
        results["fp32_map"] = baseline.get("mAP50-95")
        logger.info("FP32 mAP50-95: %.4f", results["fp32_map"] or float("nan"))

    # ── 2. Quantize + calibrate ─────────────────────────────────────────────
    calib_cfg = config["calibration"]
    section(
        logger,
        "STEP 2/7  Insert Q/DQ nodes and calibrate ranges",
        "Calibration seeds the ranges only - the model is fine-tuned next.",
    )

    calib_loader, _ = build_eval_dataloader(
        data_yaml=data_yaml,
        split=calib_cfg.get("split", "train"),
        imgsz=imgsz,
        batch_size=calib_cfg["batch_size"],
        workers=config["qat_training"].get("workers", 8),
        device=str(device),
        # Shuffled so a capped batch count samples across the dataset instead of
        # reading the first N files, which tend to be the same scene.
        shuffle=calib_cfg.get("shuffle", True),
    )

    quant_config = build_quant_config(
        algorithm=calib_cfg.get("algorithm", "max"),
        percentile=calib_cfg.get("percentile", 99.99),
        skip_patterns=tuple(
            config.get("quantization", {}).get("skip_patterns", DEFAULT_SKIP_PATTERNS)
        ),
    )

    model = quantize_model(
        model,
        quant_config,
        make_calibration_loop(calib_loader, device, calib_cfg["num_batches"]),
    )
    logger.info("\n%s", quantization_summary(model))

    # ── 3. Post-calibration (PTQ) mAP ───────────────────────────────────────
    section(
        logger,
        "STEP 3/7  Post-calibration mAP (this is the PTQ number)",
        "Everything after this point is what QAT adds on top.",
    )
    ptq = evaluate_map(model, data_yaml, **eval_kwargs)
    results["ptq_map"] = ptq.get("mAP50-95")
    logger.info("PTQ (calibrated, not retrained) mAP50-95: %.4f", results["ptq_map"] or float("nan"))

    if run_sensitivity:
        section(logger, "Optional  Per-layer sensitivity analysis")
        results["sensitivity"] = layer_sensitivity(
            model, data_yaml, top_k=config.get("quantization", {}).get("top_k", 10), **eval_kwargs
        )

    # ── 4. QAT fine-tuning ──────────────────────────────────────────────────
    section(
        logger,
        "STEP 4/7  QAT fine-tuning",
        "Fake Q/DQ is active on every forward pass.",
        "Gradients pass through it via the straight-through estimator,",
        "so the weights learn to compensate for INT8 rounding.",
    )

    overrides = build_training_overrides(
        config, run_dir=run_dir, run_name=config["output"].get("run_name", "qat")
    )
    trainer = QATTrainer(
        qat_model=model,
        overrides=overrides,
        freeze_bn_epoch=config["qat_training"].get("freeze_bn_epoch"),
    )
    trainer.train()

    trained = _load_trained_model(trainer)
    results["train_dir"] = str(getattr(trainer, "save_dir", run_dir))

    # ── 5. Final INT8 mAP ───────────────────────────────────────────────────
    section(logger, "STEP 5/7  Final simulated-INT8 mAP")
    final = evaluate_map(trained, data_yaml, **eval_kwargs)
    results["qat_map"] = final.get("mAP50-95")
    logger.info("QAT INT8 mAP50-95: %.4f", results["qat_map"] or float("nan"))

    # ── 6. Export ONNX ──────────────────────────────────────────────────────
    export_cfg = config["export"]
    section(logger, "STEP 6/7  Export ONNX with embedded Q/DQ nodes")
    onnx_path = export_qat_onnx(
        model=trained,
        onnx_path=export_cfg["onnx_path"],
        imgsz=tuple(config["model"]["input_size"]),
        opset=export_cfg.get("opset_version", 13),
        device=torch.device("cpu") if export_cfg.get("export_on_cpu", True) else device,
        dynamic_batch=export_cfg.get("dynamic_batch", False),
        fuse=export_cfg.get("fuse", True),
    )
    results["onnx_path"] = onnx_path

    # ── 7. TensorRT engine ──────────────────────────────────────────────────
    trt_cfg = config.get("tensorrt", {})
    if skip_engine:
        logger.info("Engine build skipped (--skip-engine)")
        results["engine_path"] = None
    else:
        section(logger, "STEP 7/7  Build the TensorRT INT8 engine")
        try:
            results["engine_path"] = build_engine(
                onnx_path=onnx_path,
                engine_path=trt_cfg.get("engine_path", os.path.join(run_dir, "qat_model.engine")),
                int8=trt_cfg.get("int8", True),
                fp16=trt_cfg.get("fp16", True),
                workspace_mb=trt_cfg.get("workspace_mb", 4096),
                min_batch=trt_cfg.get("min_batch", 1),
                opt_batch=trt_cfg.get("opt_batch", 1),
                max_batch=trt_cfg.get("max_batch", 1),
                verbose=trt_cfg.get("verbose", False),
            )
        except (ImportError, RuntimeError) as exc:
            logger.warning("Engine build skipped: %s", exc)
            logger.info(
                "Build it on the deployment machine with:\n"
                "  python -m src.build_trt_engine --onnx %s --output runs/qat_model.engine",
                onnx_path,
            )
            results["engine_path"] = None

    _log_summary(results, elapsed=time.time() - started)
    return results


def _log_summary(results: Dict[str, Any], elapsed: float) -> None:
    """Print the comparison that answers 'did QAT work?'."""
    fp32 = results.get("fp32_map")
    ptq = results.get("ptq_map")
    qat = results.get("qat_map")

    logger.info("")
    logger.info("=" * 68)
    logger.info("PIPELINE COMPLETE in %.1f min", elapsed / 60)
    logger.info("=" * 68)
    logger.info("  %-34s %s", "FP32 baseline mAP50-95", _fmt(fp32))
    logger.info("  %-34s %s", "PTQ (calibrated only) mAP50-95", _fmt(ptq))
    logger.info("  %-34s %s", "QAT (fine-tuned) mAP50-95", _fmt(qat))

    if ptq is not None and qat is not None:
        logger.info("  %-34s %+.4f", "QAT gain over PTQ", qat - ptq)
        if qat <= ptq:
            logger.warning(
                "QAT did not beat PTQ. Check the learning rate (too high undoes the "
                "pretrained weights), the epoch count, or whether augmentation is "
                "pushing activations outside the calibrated ranges."
            )
    if fp32 is not None and qat is not None and fp32:
        logger.info("  %-34s %.2f%%", "Accuracy retained vs FP32", qat / fp32 * 100)

    logger.info("")
    for label, key in (
        ("Training run", "train_dir"),
        ("QAT ONNX", "onnx_path"),
        ("TensorRT engine", "engine_path"),
    ):
        if results.get(key):
            logger.info("  %-16s %s", label + ":", results[key])

    logger.info("")
    logger.info("Next: copy the .engine to the DeepStream host, point")
    logger.info("deepstream/config_infer_primary.txt at it, then run:")
    logger.info("  deepstream-app -c deepstream/deepstream_app_config.txt")
    logger.info("=" * 68)


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "(skipped)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8 QAT: calibrate -> fine-tune -> export ONNX -> build engine"
    )
    parser.add_argument("--config", default="configs/qat_config.yaml", help="Path to the QAT config")
    parser.add_argument("--skip-engine", action="store_true", help="Do not build the TensorRT engine")
    parser.add_argument("--skip-baseline", action="store_true", help="Do not measure FP32 baseline mAP")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run per-layer sensitivity analysis (one validation pass per layer)",
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        skip_engine=args.skip_engine,
        skip_baseline=args.skip_baseline,
        run_sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()
