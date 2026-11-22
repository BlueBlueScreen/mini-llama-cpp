#!/usr/bin/env bash
# Verify that this port produces byte-identical output to upstream llama2.c run.c.
#
#   ./scripts/verify_upstream.sh [llama2.c-checkout] [checkpoint]
#
# - With no arguments, clones karpathy/llama2.c into a temp dir and uses
#   models/stories15M.bin (run scripts/download_model.sh first).
# - Compares both greedy decoding (temperature 0) and seeded top-p sampling.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${1:-}"
MODEL="${2:-$ROOT/models/stories15M.bin}"
PROMPT="Once upon a time"
STEPS=256

if [[ ! -f "$MODEL" ]]; then
    echo "checkpoint not found: $MODEL" >&2
    echo "run ./scripts/download_model.sh first, or pass a checkpoint path" >&2
    exit 1
fi

# Build our binary if needed.
if [[ ! -x "$ROOT/build/mini_llama" ]]; then
    cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$ROOT/build" -j
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Get an upstream checkout.
if [[ -z "$UPSTREAM" ]]; then
    git clone --depth 1 https://github.com/karpathy/llama2.c "$WORK/llama2.c"
    UPSTREAM="$WORK/llama2.c"
fi

gcc -O3 -o "$WORK/run" "$UPSTREAM/run.c" -lm

fail=0
for mode in greedy sampled; do
    if [[ "$mode" == greedy ]]; then
        ARGS=(-t 0 -n "$STEPS" -i "$PROMPT" -z "$ROOT/tokenizer.bin")
    else
        ARGS=(-t 0.8 -p 0.9 -s 42 -n "$STEPS" -i "$PROMPT" -z "$ROOT/tokenizer.bin")
    fi
    "$WORK/run" "$MODEL" "${ARGS[@]}" >"$WORK/upstream.txt" 2>/dev/null
    "$ROOT/build/mini_llama" "$MODEL" "${ARGS[@]}" >"$WORK/port.txt" 2>/dev/null
    if diff -q "$WORK/upstream.txt" "$WORK/port.txt" >/dev/null; then
        echo "PASS: $mode output is byte-identical to upstream run.c"
    else
        echo "FAIL: $mode output differs from upstream run.c" >&2
        diff "$WORK/upstream.txt" "$WORK/port.txt" | head -20 >&2
        fail=1
    fi
done
exit "$fail"
