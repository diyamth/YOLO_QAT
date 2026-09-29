"""
QAT Trainer — fine-tunes YOLOv8 with fake quantization active.

Design: subclass ultralytics' ``DetectionTrainer`` and hand it an
already-quantized model, rather than re-implementing a training loop.

The original implementation hand-rolled the loop and inherited a pile of bugs
that ultralytics solves properly:

* ``v8DetectionLoss(model)`` reads ``model.args`` as a namespace, but a model
  loaded from a ``.pt`` carries a plain ``dict`` -> ``AttributeError: 'dict'
  object has no attribute 'box'``. The trainer sets ``model.args`` correctly in
  ``set_model_attributes()``.
* Validation computed *loss*, not mAP, so there was no way to tell whether QAT
  had worked. The trainer runs a real validator every epoch.
* Warmup fell back to the wrong base LR whenever ``cosine_lr`` was off.
* No EMA, no proper scheduler handoff, no resume.

What stays QAT-specific here: forcing AMP off, freezing BN statistics late in
training, and asserting on every epoch that quantization is still live.

Why AMP is disabled
-------------------
Fake quantization computes ``clamp(round(x / scale)) * scale``. Under autocast
that arithmetic happens in fp16, whose ~3 decimal digits of mantissa are not
enough to represent an INT8 grid faithfully; rounding lands on the wrong step
and the simulated quantization stops matching what TensorRT will do. QAT
fine-tuning is short, so the lost throughput does not matter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from .qat_quantize import assert_qat_active, count_quantizers

logger = logging.getLogger("YOLO_QAT")

try:  # ultralytics moved this helper around across releases
    from ultralytics.utils.torch_utils import unwrap_model
except ImportError:  # pragma: no cover
    from ultralytics.utils.torch_utils import de_parallel as unwrap_model  # type: ignore


class QATTrainer(DetectionTrainer):
    """
    ``DetectionTrainer`` that trains a pre-quantized model.

    Args:
        qat_model: The quantized model from ``quantize_model()``.
        overrides: Ultralytics training overrides (data, epochs, batch, ...).
        freeze_bn_epoch: Epoch at which BatchNorm running statistics stop
            updating. ``None`` disables freezing.
        _callbacks: Optional ultralytics callback registry.
    """

    def __init__(
        self,
        qat_model: nn.Module,
        overrides: Dict[str, Any],
        freeze_bn_epoch: Optional[int] = None,
        _callbacks: Any = None,
    ) -> None:
        self._qat_model = qat_model
        self._freeze_bn_epoch = freeze_bn_epoch
        self._bn_frozen = False

        super().__init__(cfg=DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)

        # setup_model() short-circuits when self.model is already an nn.Module,
        # so assigning here is what stops ultralytics rebuilding an FP32 model
        # from the weights path and silently discarding the quantizers.
        self.model = qat_model

    # ── model injection ──────────────────────────────────────────────────────

    def get_model(
        self,
        cfg: Optional[str] = None,
        weights: Optional[str] = None,
        verbose: bool = True,
    ) -> nn.Module:
        """Always return the quantized model (also covers the resume path)."""
        return self._qat_model

    # ── QAT-specific training behaviour ──────────────────────────────────────

    def _model_train(self) -> None:
        """
        Put the model in train mode, then apply QAT-specific overrides.

        Ultralytics calls this at the start of every epoch, after which BN
        modules are back in training mode - so BN freezing has to be reapplied
        here rather than once up front.
        """
        super()._model_train()

        if self._freeze_bn_epoch is None or self.epoch < self._freeze_bn_epoch:
            return

        # Freeze BN running statistics. Late in QAT the quantizer ranges are
        # fixed but BN stats keep drifting, so the statistics folded into the
        # exported weights are ones the quantized weights were never trained
        # against. Freezing removes that train/export mismatch.
        frozen = 0
        for module in unwrap_model(self.model).modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                frozen += 1

        if not self._bn_frozen:
            logger.info(
                "Epoch %d: froze running statistics on %d BatchNorm layer(s)",
                self.epoch + 1,
                frozen,
            )
            self._bn_frozen = True

    def build_optimizer(self, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        """
        Build the optimizer, keeping weight decay off any quantizer parameters.

        With the default INT8 config the ranges are buffers, so there is nothing
        to fix. But learnable-range configs (LSQ-style) make ``amax`` a
        parameter, and ultralytics' grouping would sort it into the
        weight-decay group - decaying a quantization scale toward zero, which
        collapses the range. This moves any such parameter to a no-decay group.
        """
        optimizer = super().build_optimizer(*args, **kwargs)

        quant_params = {
            id(p)
            for n, p in unwrap_model(self.model).named_parameters()
            if "quantizer" in n or n.endswith("amax")
        }
        if not quant_params:
            return optimizer

        moved = []
        for group in optimizer.param_groups:
            if not group.get("weight_decay"):
                continue
            keep = [p for p in group["params"] if id(p) not in quant_params]
            moved.extend(p for p in group["params"] if id(p) in quant_params)
            group["params"] = keep

        if moved:
            optimizer.add_param_group({"params": moved, "weight_decay": 0.0})
            logger.info("Moved %d quantizer parameter(s) to a no-decay group", len(moved))
        return optimizer

    # ── guardrails ───────────────────────────────────────────────────────────

    def _setup_train(self, *args: Any, **kwargs: Any) -> None:
        # Signature varies across ultralytics releases (``world_size`` was
        # dropped in 8.4), so forward whatever we are given.
        super()._setup_train(*args, **kwargs)

        stats = assert_qat_active(unwrap_model(self.model), context="training setup")
        logger.info(
            "QAT active: %d quantizers, %d enabled, %d calibrated - "
            "fake Q/DQ runs on every forward pass, gradients flow via STE",
            stats["total"],
            stats["enabled"],
            stats["calibrated"],
        )

        if self.amp:
            # Should be unreachable: the pipeline forces amp=False. If some
            # future ultralytics re-enables it, fail rather than silently
            # training against a quantization grid that does not match INT8.
            raise RuntimeError(
                "AMP is enabled during QAT. fp16 cannot represent the INT8 "
                "rounding grid faithfully; pass amp=False."
            )

    def validate(self) -> Any:
        """Validate, confirming quantization is still live so mAP means INT8 mAP."""
        stats = count_quantizers(unwrap_model(self.model))
        if stats["enabled"] == 0:
            raise RuntimeError(
                "All quantizers are disabled at validation time - the reported "
                "mAP would be FP32, not quantized."
            )
        return super().validate()


def build_training_overrides(config: Dict[str, Any], run_dir: str, run_name: str) -> Dict[str, Any]:
    """
    Translate the project config into ultralytics training overrides.

    Args:
        config: Parsed ``qat_config.yaml``.
        run_dir: Ultralytics ``project`` directory.
        run_name: Ultralytics run ``name``.

    Returns:
        Overrides dict for :class:`QATTrainer`.
    """
    qat = config["qat_training"]

    overrides: Dict[str, Any] = {
        "model": config["model"]["weights"],
        "data": config["dataset"]["data_yaml"],
        "imgsz": config["model"]["input_size"][0],
        "epochs": qat["epochs"],
        "batch": qat["batch_size"],
        "lr0": qat["learning_rate"],
        "lrf": qat.get("final_lr_ratio", 0.01),
        "weight_decay": qat.get("weight_decay", 0.0005),
        "warmup_epochs": qat.get("warmup_epochs", 1),
        "optimizer": qat.get("optimizer", "AdamW"),
        "cos_lr": qat.get("cosine_lr", True),
        "workers": qat.get("workers", 8),
        "device": qat.get("device", "0"),
        "project": run_dir,
        "name": run_name,
        "exist_ok": True,
        "pretrained": False,  # weights already live in the quantized model
        "val": True,
        "plots": qat.get("plots", True),
        "save": True,
        # ── non-negotiable for QAT ──
        "amp": False,  # fp16 cannot represent the INT8 grid faithfully
        "compile": False,  # torch.compile would trace away the quantizers
    }

    # QAT is a short fine-tune from converged weights. Heavy augmentation
    # fights that: it pushes activations into ranges the calibration never saw,
    # so the fixed quantizer ranges clip data the model now depends on.
    if qat.get("reduce_augmentation", True):
        overrides.update(
            {
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "erasing": 0.0,
                "scale": qat.get("aug_scale", 0.3),
                "translate": qat.get("aug_translate", 0.1),
                "fliplr": qat.get("aug_fliplr", 0.5),
            }
        )

    # Anything under qat_training.extra_args wins, for one-off experiments.
    overrides.update(qat.get("extra_args", {}) or {})
    return overrides
