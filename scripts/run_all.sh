#!/usr/bin/env bash
# One-shot QAT pipeline: calibrate -> fine-tune -> export ONNX -> build engine.
#
# Usage:
#   ./scripts/run_all.sh [config] [extra args passed to the pipeline]
#
# Examples:
#   ./scripts/run_all.sh
#   ./scripts/run_all.sh configs/qat_config.yaml --skip-engine
#   ./scripts/run_all.sh configs/qat_config.yaml --sensitivity

set -euo pipefail

CONFIG="${1:-configs/qat_config.yaml}"
shift || true

cd "$(dirname "$0")/.."

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

echo "==> Checking the quantization backend"
python -c "import modelopt.torch.quantization" 2>/dev/null || {
  echo "nvidia-modelopt is not installed. Run: pip install -r requirements.txt" >&2
  exit 1
}

echo "==> Running integrity tests"
python -m pytest tests/ -q

echo "==> Running the QAT pipeline with ${CONFIG}"
python -m src.run_qat_pipeline --config "${CONFIG}" "$@"
