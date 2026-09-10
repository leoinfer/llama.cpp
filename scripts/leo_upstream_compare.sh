#!/usr/bin/env bash
set -euo pipefail

# Compare a clean upstream checkout with the current fork without copying model
# data. Both benchmark invocations are serialized under the shared GPU lock.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:?set MODEL to a real qwen4exp GGUF file}
OUT=${OUT:-"$REPO/../overnight-flash-next-campaign/03_BASELINE/llama-compare-$(date -u +%Y%m%dT%H%M%SZ)"}
PROMPTS=${PROMPTS:-128,512}
NGEN=${NGEN:-128}
REPETITIONS=${REPETITIONS:-3}
UPSTREAM_REF=${UPSTREAM_REF:-upstream/master}
UPSTREAM_TREE=${UPSTREAM_TREE:-/tmp/leo-llama-upstream-tree}
UPSTREAM_BUILD=${UPSTREAM_BUILD:-/tmp/leo-llama-upstream-build}
FORK_BUILD=${FORK_BUILD:-/tmp/leo-llama-fork-build}

if [[ ! -f "$MODEL" ]]; then
    printf 'model must be a local regular file: %s\n' "$MODEL" >&2
    exit 2
fi
MODEL_BYTES=$(stat -c '%s' "$MODEL")
if (( MODEL_BYTES < 1000000000 )); then
    printf 'refusing fixture-sized model (%s bytes)\n' "$MODEL_BYTES" >&2
    exit 2
fi
mkdir -p "$OUT"
git -C "$REPO" fetch --quiet upstream master
UPSTREAM_SHA=$(git -C "$REPO" rev-parse "$UPSTREAM_REF")
FORK_SHA=$(git -C "$REPO" rev-parse HEAD)
if [[ -e "$UPSTREAM_TREE/.git" ]]; then
    git -C "$REPO" worktree remove --force "$UPSTREAM_TREE"
fi
git -C "$REPO" worktree add --detach "$UPSTREAM_TREE" "$UPSTREAM_SHA" >/dev/null
trap 'git -C "$REPO" worktree remove --force "$UPSTREAM_TREE" >/dev/null 2>&1 || true' EXIT

configure() {
    local source=$1 build=$2
    cmake -S "$source" -B "$build" -G Ninja \
        -DGGML_VULKAN=ON -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_APP=ON -DLLAMA_OPENSSL=OFF >/dev/null
    ninja -C "$build" -j8 >/dev/null
}
configure "$UPSTREAM_TREE" "$UPSTREAM_BUILD"
configure "$REPO" "$FORK_BUILD"

/home/leo/bin/research-machine-bridge verify --pretty > "$OUT/preflight.json"
cat > "$OUT/manifest.json" <<EOF
{
  "schema": "flash-next.llama-upstream-compare.v1",
  "model": "$MODEL",
  "model_bytes": $MODEL_BYTES,
  "upstream_sha": "$UPSTREAM_SHA",
  "fork_sha": "$FORK_SHA",
  "backend": "Vulkan0",
  "gpu_lock": "/tmp/leo-gpu0.lock",
  "mtp": false,
  "prompt_lengths": "$PROMPTS",
  "n_gen": $NGEN,
  "repetitions": $REPETITIONS
}
EOF

exec 9>/tmp/leo-gpu0.lock
if ! flock -n 9; then
    printf 'GPU lock busy: /tmp/leo-gpu0.lock\n' >&2
    exit 75
fi
for label in upstream fork; do
    if [[ "$label" == upstream ]]; then
        bench="$UPSTREAM_BUILD/bin/llama-bench"
    else
        bench="$FORK_BUILD/bin/llama-bench"
    fi
    GGML_VK_VISIBLE_DEVICES=0 python3 "$REPO/scripts/leo_exec_measure.py" "$bench" \
        --model "$MODEL" --output json --repetitions "$REPETITIONS" \
        --n-prompt "$PROMPTS" --n-gen "$NGEN" --batch-size 512 --ubatch-size 512 \
        --n-gpu-layers 999 --device Vulkan0 --flash-attn on --load-mode mmap \
        --lazy-mode on --cache-type-k f16 --cache-type-v f16 \
        > "$OUT/$label.json" 2> "$OUT/$label.time"
done
printf '%s\n' "$OUT"
