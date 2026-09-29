"""
Integrity tests — prove the pipeline does true QAT, not PTQ.

The original implementation *claimed* QAT everywhere in its docstrings while
inserting zero quantizers, and nothing in the repo could have caught that. The
difference between QAT and PTQ is not a comment; it is four testable facts:

1. Q/DQ nodes exist and carry calibrated ranges.
2. They change the forward pass (fake quantization is actually applied).
3. Gradients flow *through* them to the weights (the straight-through
   estimator works) - this is what "aware" means.
4. An optimizer step moves the weights while quantization is active, so the
   weights adapt to quantization noise. PTQ cannot do this by construction.

Plus the deployment contract: the exported ONNX carries Q/DQ nodes and exactly
one output.

Run:
    pytest tests/ -v
    python tests/test_qat_integrity.py     # no pytest needed
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qat_quantize import (  # noqa: E402
    MODELOPT_AVAILABLE,
    assert_qat_active,
    build_quant_config,
    count_quantizers,
    make_calibration_loop,
    quantization_disabled,
    quantize_model,
)

pytestmark = pytest.mark.skipif(
    not MODELOPT_AVAILABLE, reason="nvidia-modelopt is not installed"
)

IMGSZ = 160
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "yolov8s.pt")


def _flatten(output):
    """
    Collect every tensor from a model output.

    In train mode the detect head returns a dict keyed by head
    (``one2many``/``one2one``) in ultralytics >= 8.4, a list of feature maps in
    older releases, and a bare tensor in export mode. Flattening covers all of
    them so these tests do not break on an ultralytics upgrade.
    """
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, dict):
        return [t for value in output.values() for t in _flatten(value)]
    if isinstance(output, (list, tuple)):
        return [t for value in output for t in _flatten(value)]
    return []


def _scalar_loss(output) -> torch.Tensor:
    """A simple differentiable scalar over every output tensor."""
    tensors = [t for t in _flatten(output) if t.is_floating_point()]
    assert tensors, "model produced no floating-point output tensors"
    return sum(t.float().pow(2).mean() for t in tensors)


@pytest.fixture(scope="module")
def quantized_model():
    """A calibrated, quantization-aware YOLOv8s on CPU."""
    from ultralytics import YOLO

    model = YOLO(WEIGHTS).model.float().eval()

    # YOLO() hands back an inference-mode model with requires_grad=False on
    # every parameter. Ultralytics' trainer re-enables them during _setup_train;
    # mirror that here so the gradient tests exercise the training path.
    for parameter in model.parameters():
        if parameter.dtype.is_floating_point:
            parameter.requires_grad_(True)

    batches = [torch.rand(1, 3, IMGSZ, IMGSZ) for _ in range(2)]
    return quantize_model(
        model,
        build_quant_config(algorithm="max"),
        make_calibration_loop(batches, torch.device("cpu"), len(batches)),
    )


def test_quantizers_inserted_and_calibrated(quantized_model):
    """Q/DQ nodes exist and every enabled one has a range."""
    stats = count_quantizers(quantized_model)
    assert stats["total"] > 0, "no quantizers were inserted"
    assert stats["enabled"] > 0, "every quantizer is disabled"
    assert stats["calibrated"] == stats["enabled"], "some enabled quantizer has no range"
    assert_qat_active(quantized_model, context="integrity test")


def test_dfl_layer_excluded(quantized_model):
    """The DFL conv stays FP32 - INT8 there costs accuracy and saves nothing."""
    enabled_dfl = [
        name
        for name, q in __import__(
            "src.qat_quantize", fromlist=["iter_quantizers"]
        ).iter_quantizers(quantized_model)
        if "dfl" in name and q.is_enabled
    ]
    assert not enabled_dfl, f"DFL quantizers should be disabled, found: {enabled_dfl}"


def test_fake_quantization_changes_the_forward_pass(quantized_model):
    """
    Quantized and FP32 forward passes must differ.

    If they matched, the quantizers would be pass-throughs and every downstream
    'QAT' claim would be false - which is exactly the bug this repo shipped with.
    """
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    quantized_model.eval()

    with torch.no_grad():
        quantized_out = _flatten(quantized_model(x))[0]
        with quantization_disabled(quantized_model):
            fp32_out = _flatten(quantized_model(x))[0]

    delta = (quantized_out - fp32_out).abs().max().item()
    assert delta > 0, "quantized and FP32 outputs are identical - fake quant is inactive"

    # Sanity: the difference is quantization noise, not a broken graph.
    scale = fp32_out.abs().max().item()
    assert delta < scale, f"quantization changed the output by {delta:.4f} vs scale {scale:.4f}"


def test_context_manager_restores_quantizer_state(quantized_model):
    """quantization_disabled must not permanently flatten the model."""
    before = count_quantizers(quantized_model)
    with quantization_disabled(quantized_model):
        during = count_quantizers(quantized_model)
    after = count_quantizers(quantized_model)

    assert during["enabled"] == 0, "quantizers still enabled inside the context"
    assert after["enabled"] == before["enabled"], "quantizer state was not restored"


def test_gradients_flow_through_quantizers(quantized_model):
    """
    The straight-through estimator must pass gradients to the weights.

    Fake quantization rounds, and round() has zero gradient almost everywhere.
    The STE substitutes an identity gradient so training can proceed. Without
    it every weight gradient would be zero and 'training' would be a no-op --
    the model would never adapt to quantization, which is the whole point.
    """
    model = quantized_model
    model.train()
    model.zero_grad(set_to_none=True)

    loss = _scalar_loss(model(torch.rand(2, 3, IMGSZ, IMGSZ)))
    loss.backward()

    conv_grads = [
        (name, p.grad)
        for name, p in model.named_parameters()
        if p.requires_grad and name.endswith("weight") and p.grad is not None and p.ndim == 4
    ]
    assert conv_grads, "no conv weight received a gradient"

    nonzero = [name for name, g in conv_grads if g.abs().sum().item() > 0]
    assert nonzero, "every conv gradient was zero - the STE is not passing gradients"
    # A healthy backward reaches most of the network, not one stray layer.
    assert len(nonzero) > len(conv_grads) // 2, (
        f"only {len(nonzero)}/{len(conv_grads)} conv layers got a gradient"
    )
    model.zero_grad(set_to_none=True)


def test_optimizer_step_updates_weights_under_quantization(quantized_model):
    """
    Weights must actually move while quantization is active.

    This is the line between QAT and PTQ: PTQ freezes the weights and only
    picks scales; QAT keeps training them against simulated INT8 error.
    """
    model = quantized_model
    model.train()

    target = next(
        p for n, p in model.named_parameters() if n.endswith("weight") and p.ndim == 4 and p.requires_grad
    )
    before = target.detach().clone()

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    loss = _scalar_loss(model(torch.rand(2, 3, IMGSZ, IMGSZ)))
    loss.backward()
    optimizer.step()

    moved = (target.detach() - before).abs().max().item()
    assert moved > 0, "weights did not change after an optimizer step - this is PTQ, not QAT"
    model.zero_grad(set_to_none=True)


def test_export_produces_qdq_onnx_with_single_output(quantized_model):
    """
    The exported graph must carry Q/DQ and exactly one output.

    Without Q/DQ, TensorRT silently builds an FP32 engine and reports success.
    More than one output and the DeepStream bbox parser cannot bind.
    """
    import onnx

    from src.qat_export import export_qat_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_qat_onnx(
            model=quantized_model,
            onnx_path=os.path.join(tmpdir, "qat.onnx"),
            imgsz=(IMGSZ, IMGSZ),
            opset=13,
            device=torch.device("cpu"),
        )
        graph = onnx.load(path)

        quantize_nodes = [n for n in graph.graph.node if n.op_type == "QuantizeLinear"]
        dequantize_nodes = [n for n in graph.graph.node if n.op_type == "DequantizeLinear"]

        assert quantize_nodes, "exported ONNX has no QuantizeLinear nodes"
        assert dequantize_nodes, "exported ONNX has no DequantizeLinear nodes"
        assert len(graph.graph.output) == 1, (
            f"expected 1 output for DeepStream, got {[o.name for o in graph.graph.output]}"
        )
        assert graph.graph.output[0].name == "output0"


def test_export_refuses_an_unquantized_model():
    """Exporting a plain FP32 model must fail loudly, not emit a fake 'QAT' ONNX."""
    from ultralytics import YOLO

    from src.qat_export import export_qat_onnx

    model = YOLO(WEIGHTS).model.float().eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="[Nn]o quantizers"):
            export_qat_onnx(
                model=model,
                onnx_path=os.path.join(tmpdir, "bad.onnx"),
                imgsz=(IMGSZ, IMGSZ),
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-header"]))
