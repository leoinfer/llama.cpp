#!/usr/bin/env bash
set -euo pipefail

# Stock-only baseline harness. It deliberately has no speculative/MTP flags and
# refuses to invent a fallback model when the requested real artifact is absent.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-/tmp/leo-llama-build/bin/llama-bench}
MODEL=${MODEL:?set MODEL to a real qwen4exp GGUF file}
OUT=${OUT:-"$ROOT/../overnight-flash-next-campaign/03_BASELINE/llama-baseline-0-$(date -u +%Y%m%dT%H%M%SZ)"}
PROMPTS=${PROMPTS:-128,512,2048,8192,32768,131072,262144}
REPETITIONS=${REPETITIONS:-3}
NGEN=${NGEN:-128}

if [[ ! -x "$BENCH" ]]; then
    printf 'missing benchmark binary: %s\n' "$BENCH" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    printf 'model must be a local regular file: %s\n' "$MODEL" >&2
    exit 2
fi
mkdir -p "$OUT"
MODEL_BYTES=$(stat -c '%s' "$MODEL")
if (( MODEL_BYTES < 1000000000 )); then
    printf 'refusing fixture-sized model (%s bytes); provide a real qwen4exp GGUF\n' "$MODEL_BYTES" >&2
    exit 2
fi

# llama.cpp does not use the HAR lock internally; hold the real shared lock for
# the preflight and complete workload so no other Vulkan lane can overlap it.
exec 9>/tmp/leo-gpu0.lock
if ! flock -n 9; then
    printf 'GPU lock busy: /tmp/leo-gpu0.lock\n' >&2
    exit 75
fi

/home/leo/bin/research-machine-bridge verify --pretty > "$OUT/preflight.json"
cat > "$OUT/manifest.json" <<EOF
{
  "schema": "flash-next.llama-stock-baseline-run.v1",
  "model": "$(printf '%s' "$MODEL" | sed 's/\\/\\\\/g; s/"/\\"/g')",
  "model_bytes": $MODEL_BYTES,
  "backend": "Vulkan0",
  "gpu_visibility": "GGML_VK_VISIBLE_DEVICES=0",
  "gpu_lock": "/tmp/leo-gpu0.lock",
  "mtp": false,
  "load_mode": "mmap",
  "lazy_mode": "on",
  "flash_attention": true,
  "repetitions": $REPETITIONS,
  "n_gen": $NGEN
}
EOF
GGML_VK_VISIBLE_DEVICES=0 python3 "$ROOT/scripts/leo_exec_measure.py" "$BENCH" \
    --model "$MODEL" \
    --output json \
    --repetitions "$REPETITIONS" \
    --n-prompt "$PROMPTS" \
    --n-gen "$NGEN" \
    --batch-size 512 \
    --ubatch-size 512 \
    --n-gpu-layers 999 \
    --device Vulkan0 \
    --flash-attn on \
    --load-mode mmap \
    --lazy-mode on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    > "$OUT/llama-bench.json" 2> "$OUT/llama-bench.time"
printf '%s\n' "$OUT"
