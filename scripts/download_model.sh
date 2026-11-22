#!/usr/bin/env bash
# Download a TinyStories checkpoint (legacy llama2.c v0 format) into models/.
#
#   ./scripts/download_model.sh          # stories15M.bin (~58 MB)
#   ./scripts/download_model.sh 42M      # stories42M.bin (~167 MB)
#   ./scripts/download_model.sh 110M     # stories110M.bin (~438 MB)
#
# Set HF_ENDPOINT to use a mirror, e.g. HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

SIZE="${1:-15M}"
case "$SIZE" in
    15M|42M|110M) ;;
    *) echo "usage: $0 [15M|42M|110M]" >&2; exit 1 ;;
esac

DEST="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DEST"

BASE="${HF_ENDPOINT:-https://huggingface.co}"
URL="${BASE%/}/karpathy/tinyllamas/resolve/main/stories${SIZE}.bin"
OUT="$DEST/stories${SIZE}.bin"

if [[ -f "$OUT" ]]; then
    echo "already exists: $OUT"
    exit 0
fi

echo "downloading stories${SIZE}.bin from $URL"
curl -L --fail --progress-bar -o "$OUT.part" "$URL"
mv "$OUT.part" "$OUT"
echo "saved: $OUT"
