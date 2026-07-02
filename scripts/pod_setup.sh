#!/usr/bin/env bash
# Bootstrap vLLM on a bare GPU pod (RunPod / Vast container instances, where
# docker-compose is unavailable). Gateway + observability run elsewhere via
# docker compose, pointed at this pod (split mode).
#
# Usage, from the repo root on the pod:
#   bash scripts/pod_setup.sh lora    # runtime LoRA hot-swap serving (default)
#   bash scripts/pod_setup.sh perf    # FP8 + ngram speculative decoding
#
# Expose ports 8000 (vLLM OpenAI API + /metrics) and 9835 (GPU exporter).

set -euo pipefail

MODE="${1:-lora}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Coder-3B}"

echo "==> Installing vLLM 0.7.3 (mode: ${MODE}, model: ${MODEL_ID})"
# Pin transformers: vLLM 0.7.3's tokenizer shim breaks on transformers>=5
# (Qwen2Tokenizer.all_special_tokens_extended was removed). 4.49.x is the
# last line vLLM 0.7.3 was released against.
pip install -q "vllm==0.7.3" "transformers==4.49.0" "hf_transfer"

echo "==> Starting GPU metrics exporter on :9835"
nohup python scripts/gpu_metrics_exporter.py --port 9835 >gpu_exporter.log 2>&1 &

# ngram speculative decoding and runtime LoRA loading require the v0 engine in 0.7.x
export VLLM_USE_V1=0
export HF_HUB_ENABLE_HF_TRANSFER=1

COMMON_ARGS=(
  --host 0.0.0.0
  --port 8000
  --quantization fp8
  --max-model-len 4096
  --gpu-memory-utilization 0.90
  --enable-prefix-caching
  --disable-log-requests
)

case "$MODE" in
  perf)
    echo "==> Serving ${MODEL_ID}: FP8 + ngram speculative decoding"
    exec vllm serve "$MODEL_ID" "${COMMON_ARGS[@]}" \
      --speculative-model "[ngram]" \
      --num-speculative-tokens 5 \
      --ngram-prompt-lookup-max 4
    ;;
  lora)
    echo "==> Serving ${MODEL_ID}: FP8 + runtime LoRA hot-swap"
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    exec vllm serve "$MODEL_ID" "${COMMON_ARGS[@]}" \
      --enable-lora \
      --max-lora-rank 32 \
      --max-loras 4
    ;;
  *)
    echo "Unknown mode '$MODE' (expected: perf | lora)" >&2
    exit 1
    ;;
esac
