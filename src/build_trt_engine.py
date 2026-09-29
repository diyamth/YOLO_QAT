"""
TensorRT engine builder — build an INT8 engine from a QAT ONNX graph.

How Q/DQ actually reaches TensorRT
----------------------------------
There is no flag to "turn on" QAT. When the parsed ONNX contains
``QuantizeLinear``/``DequantizeLinear`` nodes, TensorRT switches to **explicit
quantization** on its own: it reads the scale from each Q/DQ pair, propagates
them through the graph, and fuses the pairs into the neighbouring convolutions.

The original implementation set ``NetworkDefinitionCreationFlag.EXPLICIT_PRECISION``
and logged that this made TensorRT use the Q/DQ nodes. That flag is a removed
TensorRT 7-era mechanism unrelated to Q/DQ; it was behind a ``hasattr`` guard, so
on any modern TensorRT it silently did nothing. Dropped here.

``BuilderFlag.INT8`` is still set: it tells the builder INT8 tactics may be
selected. ``BuilderFlag.FP16`` is set alongside it by default so that layers
TensorRT cannot run in INT8 fall back to FP16 rather than FP32.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("YOLO_QAT")


def _check_onnx_has_qdq(onnx_path: str) -> int:
    """
    Count Q/DQ nodes, warning loudly if there are none.

    Without them TensorRT builds a *valid* FP32/FP16 engine and reports success,
    so this is the last place the mistake is cheap to catch.
    """
    try:
        import onnx
    except ImportError:  # pragma: no cover
        logger.warning("onnx not installed; skipping Q/DQ pre-check")
        return -1

    graph = onnx.load(onnx_path)
    counts = collections.Counter(node.op_type for node in graph.graph.node)
    qdq = counts["QuantizeLinear"] + counts["DequantizeLinear"]

    if qdq == 0:
        logger.warning(
            "%s has no Q/DQ nodes. TensorRT will build an FP32/FP16 engine and "
            "report success - it will NOT be INT8.",
            onnx_path,
        )
    else:
        logger.info("ONNX carries %d Q/DQ node(s) - explicit INT8 quantization", qdq)
    return qdq


def build_engine(
    onnx_path: str,
    engine_path: str,
    int8: bool = True,
    fp16: bool = True,
    workspace_mb: int = 4096,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    verbose: bool = False,
) -> str:
    """
    Build a TensorRT engine from a QAT ONNX file.

    Args:
        onnx_path: Path to the QAT ONNX (with Q/DQ nodes).
        engine_path: Output engine path.
        int8: Allow INT8 tactics. Required for the Q/DQ scales to be used.
        fp16: Also allow FP16, so non-quantized layers avoid falling back to FP32.
        workspace_mb: Workspace memory pool, in MB.
        min_batch: Minimum batch for the optimization profile.
        opt_batch: Batch size TensorRT tunes tactics for. Set this to the batch
            you actually serve - DeepStream's ``batch-size``.
        max_batch: Maximum batch for the optimization profile.
        verbose: Verbose TensorRT logging.

    Returns:
        The engine path written.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT Python bindings not found. Install the build matching your "
            "CUDA/TensorRT version, e.g. `pip install tensorrt`, or use the "
            "packages that ship with DeepStream."
        ) from exc

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)) or ".", exist_ok=True)
    _check_onnx_has_qdq(onnx_path)

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    logger.info("TensorRT %s", trt.__version__)

    builder = trt.Builder(trt_logger)

    # EXPLICIT_BATCH is implicit and deprecated from TensorRT 10 onward; only
    # pass it where it still exists.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            for error in errors:
                logger.error("ONNX parser: %s", error)
            raise RuntimeError(f"Failed to parse {onnx_path}:\n  " + "\n  ".join(errors))

    logger.info(
        "Parsed ONNX: %d layers, %d input(s), %d output(s)",
        network.num_layers,
        network.num_inputs,
        network.num_outputs,
    )

    config = builder.create_builder_config()

    # TensorRT 8.4 renamed max_workspace_size to the memory-pool API.
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    else:  # pragma: no cover - TensorRT < 8.4
        config.max_workspace_size = workspace_mb * (1 << 20)

    if int8:
        if not builder.platform_has_fast_int8:
            logger.warning("This GPU has no fast INT8 support; the engine may not speed up.")
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 enabled - scales come from the ONNX Q/DQ nodes")
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled - fallback precision for non-INT8 layers")

    _add_optimization_profile(
        builder, config, network, trt, min_batch, opt_batch, max_batch
    )

    logger.info("Building engine (this can take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "TensorRT failed to build the engine. Re-run with verbose=True for the "
            "builder log."
        )

    with open(engine_path, "wb") as handle:
        handle.write(serialized)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info("Engine written: %s (%.1f MB)", engine_path, size_mb)
    _inspect_engine(engine_path, trt, trt_logger)
    return engine_path


def _add_optimization_profile(
    builder: Any,
    config: Any,
    network: Any,
    trt: Any,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    """
    Add a profile only when the network actually has dynamic dimensions.

    A fixed-shape network needs no profile, and adding one for a static input is
    a build error rather than a no-op.
    """
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    if not any(any(d < 0 for d in tensor.shape) for tensor in inputs):
        logger.info("Static input shape %s - no optimization profile needed", tuple(inputs[0].shape))
        return

    if not (min_batch <= opt_batch <= max_batch):
        raise ValueError(
            f"Batch bounds must satisfy min <= opt <= max, got "
            f"{min_batch}/{opt_batch}/{max_batch}"
        )

    profile = builder.create_optimization_profile()
    for tensor in inputs:
        shape = list(tensor.shape)
        # Only the batch dimension is treated as dynamic; a dynamic spatial
        # dimension would need explicit bounds we cannot guess.
        if any(d < 0 for d in shape[1:]):
            raise RuntimeError(
                f"Input {tensor.name!r} has dynamic non-batch dims {shape}. Re-export "
                "with a fixed resolution."
            )
        rest = shape[1:]
        profile.set_shape(
            tensor.name,
            tuple([min_batch] + rest),
            tuple([opt_batch] + rest),
            tuple([max_batch] + rest),
        )
        logger.info(
            "Profile %s: min=%s opt=%s max=%s",
            tensor.name,
            [min_batch] + rest,
            [opt_batch] + rest,
            [max_batch] + rest,
        )
    config.add_optimization_profile(profile)


def _inspect_engine(engine_path: str, trt: Any, trt_logger: Any) -> None:
    """Deserialize the engine and log its bindings, as a build sanity check."""
    try:
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            logger.warning("Engine built but could not be deserialized for inspection")
            return

        if hasattr(engine, "num_io_tensors"):  # TensorRT 10
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                logger.info(
                    "  %s %s %s %s",
                    "input " if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output",
                    name,
                    engine.get_tensor_shape(name),
                    engine.get_tensor_dtype(name),
                )
        else:  # pragma: no cover - TensorRT 8
            for i in range(engine.num_bindings):
                logger.info(
                    "  %s %s %s %s",
                    "input " if engine.binding_is_input(i) else "output",
                    engine.get_binding_name(i),
                    engine.get_binding_shape(i),
                    engine.get_binding_dtype(i),
                )
    except Exception as exc:  # noqa: BLE001 - inspection must never fail the build
        logger.warning("Engine inspection skipped: %s", exc)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build a TensorRT INT8 engine from a QAT ONNX")
    parser.add_argument("--onnx", required=True, help="Path to the QAT ONNX file")
    parser.add_argument("--output", required=True, help="Output engine path")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 tactics")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 fallback")
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace MB")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="Verbose TensorRT logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        int8=not args.no_int8,
        fp16=not args.no_fp16,
        workspace_mb=args.workspace,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
