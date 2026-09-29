# YOLOv8 Quantization-Aware Training → TensorRT INT8 → DeepStream

Quantization-aware training for YOLOv8 using **NVIDIA TensorRT Model Optimizer**,
exporting an ONNX graph with embedded Q/DQ nodes and building a native INT8
TensorRT engine for DeepStream.

## QAT vs PTQ, concretely

| | PTQ | QAT (this project) |
|---|---|---|
| Training | none | fine-tunes with fake quantization active |
| Quantization error | absorbed at inference | learned against during training |
| Ranges | calibrated, then frozen | calibrated, then weights adapt to them |
| TensorRT input | ONNX + calibration cache | ONNX with Q/DQ nodes, no cache |

The pipeline reports **both** numbers, so the distinction is measured rather
than asserted. Step 3 is post-calibration mAP — that *is* the PTQ result. Step 5
is mAP after fine-tuning. The gap between them is what QAT bought you, and the
run warns if it is not positive.

## Pipeline

```
FP32 weights
  ├── [1] baseline mAP ............ what we are trying not to lose
  ├── [2] insert Q/DQ + calibrate . ranges seeded from real data
  ├── [3] post-calibration mAP .... the PTQ number
  ├── [4] QAT fine-tune ........... weights adapt to quantization noise
  ├── [5] final INT8 mAP .......... QAT vs PTQ is the payoff
  ├── [6] export ONNX with Q/DQ
  └── [7] build TensorRT INT8 engine
```

## Quick start

```bash
pip install -r requirements.txt

# point dataset.data_yaml at your data, set num classes, device, epochs
vim configs/qat_config.yaml

# prove the setup does real QAT before spending GPU hours
pytest tests/ -v

# full run
python -m src.run_qat_pipeline --config configs/qat_config.yaml

# or one-shot
./scripts/run_all.sh
```

Useful flags:

```bash
--skip-engine      # stop after ONNX (e.g. build the engine on the target host)
--skip-baseline    # skip the FP32 measurement
--sensitivity      # per-layer analysis: which layers should stay FP32
```

Build an engine separately, on the deployment machine:

```bash
python -m src.build_trt_engine --onnx runs/qat_model.onnx \
                               --output runs/qat_model.engine --opt-batch 1
```

## Requirements

* **ultralytics >= 8.4** — hard requirement. Its checkpoint writer moves
  ModelOpt quantization into a sidecar entry (runtime-generated quantized
  classes cannot be pickled) and restores it on load, and its `fuse()` rescales
  weight-quantizer ranges when folding BatchNorm. Older releases fail to save a
  quantized model, or silently lose the quantization.
* **nvidia-modelopt** — the quantization backend.
* **TensorRT** — only for step 7. Install the build matching the CUDA/TensorRT
  on your deployment host; DeepStream ships its own.

QAT itself needs a CUDA GPU. The quantization, export and test paths run on CPU,
which is how the integrity suite works on a laptop.

## Verifying it actually does QAT

`tests/test_qat_integrity.py` checks the things that distinguish QAT from a
model that merely claims to be quantized:

* Q/DQ nodes exist and carry calibrated ranges
* the DFL conv is excluded from quantization
* fake quantization measurably changes the forward pass
* **gradients flow through the quantizers** (the straight-through estimator)
* **an optimizer step moves the weights while quantization is active** — PTQ
  cannot do this by construction
* the exported ONNX carries Q/DQ nodes and exactly one output
* exporting an unquantized model raises instead of emitting a fake "QAT" ONNX

```bash
pytest tests/ -v          # 8 passed
```

## Tuning notes

**Learning rate.** This is a fine-tune of converged weights. `1e-5` to `1e-4`.
A large LR destroys the pretrained weights faster than quantization noise is
ever learned.

**Augmentation.** Reduced by default. Heavy augmentation drives activations into
ranges calibration never saw, and the fixed quantizer ranges then clip data the
model depends on.

**Which layers to exclude.** Run `--sensitivity`. It disables one layer's
quantizers at a time and re-measures mAP, ranking layers by how much accuracy
leaving them in FP32 recovers. Feed the worst offenders back into
`quantization.skip_patterns` and retrain. It costs one validation pass per
layer, so point it at a small split.

**Calibration.** `algorithm: max` is the NVIDIA default and a good baseline. Try
`percentile` (with `percentile: 99.99`) when a few outlier activations are
stretching the range and coarsening every step.

## Deploying to DeepStream

**You must build a custom bbox parser.** YOLOv8 emits a single raw tensor of
shape `(batch, 4 + num_classes, anchors)` — `(1, 84, 8400)` for COCO — that is
neither decoded nor NMS'd. No built-in nvinfer parser understands that layout,
so without one the pipeline runs cleanly and shows **zero detections**.

```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo
cd DeepStream-Yolo && CUDA_VER=12.2 make -C nvdsinfer_custom_impl_Yolo
```

Then uncomment `parse-bbox-func-name` and `custom-lib-path` in
`deepstream/config_infer_primary.txt`, and check that `num-detected-classes`,
`labels.txt` and `infer-dims` all match your model.

```bash
deepstream-app -c deepstream/deepstream_app_config.txt
```

## Project layout

```
configs/qat_config.yaml       all configuration
src/
  run_qat_pipeline.py         orchestrator (the 7 steps above)
  qat_quantize.py             Q/DQ insertion, calibration, quantizer control
  qat_trainer.py              QAT fine-tuning (ultralytics DetectionTrainer)
  qat_validate.py             mAP, FP32-vs-INT8 comparison, sensitivity
  qat_export.py               ONNX export with Q/DQ
  qat_data.py                 calibration/eval dataloaders
  build_trt_engine.py         TensorRT INT8 engine builder
  utils.py                    config, logging, device
tests/test_qat_integrity.py   proves QAT is real, not claimed
deepstream/                   nvinfer + deepstream-app configs, labels
scripts/run_all.sh            one-shot pipeline
```

## Gotchas

* **Never run onnx-simplifier on the exported ONNX.** It folds the Q/DQ pairs
  away and silently leaves an FP32 graph. `onnx-simplifier` is deliberately
  absent from `requirements.txt`.
* **A missing-Q/DQ ONNX still builds a working engine.** TensorRT reports
  success and produces a perfectly good FP32 engine. `qat_export.py` raises
  rather than let that through.
* **AMP is forced off during QAT.** fp16 cannot represent the INT8 rounding grid
  faithfully, so the simulated quantization stops matching what TensorRT does.
* **`quantize` is not a flag you pass TensorRT.** Explicit quantization is
  triggered by Q/DQ nodes being present in the ONNX, nothing else.
