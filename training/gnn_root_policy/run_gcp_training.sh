#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/ember-gnn}"
DATA_URL="${DATA_URL:-https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/test80-2022-08-aug-16tb7p.v6-dd.min.binpack?download=true}"
SAMPLES="${SAMPLES:-5000000}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
HIDDEN="${HIDDEN:-96}"
LAYERS="${LAYERS:-3}"
WORKERS="${WORKERS:-2}"
VALID_BATCHES="${VALID_BATCHES:-64}"

mkdir -p "$ROOT"/{data,samples,artifacts,logs}
cd "$ROOT"

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install numpy tqdm onnx onnxruntime 'torch==2.4.1+cu124' \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple

BINPACK="$ROOT/data/test80.binpack"
if [ ! -s "$BINPACK" ]; then
  wget -O "$BINPACK" "$DATA_URL"
fi

EXTRACTOR="$ROOT/src/extract_sfbinpack"
TRAINER="$ROOT/src/train_root_policy.py"
SAMPLE_FILE="$ROOT/samples/root-policy-${SAMPLES}.bin"
RUN_DIR="$ROOT/artifacts/run-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

if [ ! -s "$SAMPLE_FILE" ]; then
  cargo run --release --manifest-path "$EXTRACTOR/Cargo.toml" -- \
    --input "$BINPACK" \
    --output "$SAMPLE_FILE" \
    --max-samples "$SAMPLES" \
    2>&1 | tee "$RUN_DIR/extract.log"
fi

python "$TRAINER" \
  --samples "$SAMPLE_FILE" \
  --output-dir "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --hidden "$HIDDEN" \
  --layers "$LAYERS" \
  --valid-batches "$VALID_BATCHES" \
  2>&1 | tee "$RUN_DIR/train.log"

sha256sum "$RUN_DIR/root_policy.onnx" > "$RUN_DIR/root_policy.onnx.sha256"
tar -C "$ROOT/artifacts" -czf "$ROOT/artifacts/$(basename "$RUN_DIR").tar.gz" "$(basename "$RUN_DIR")"
echo "$RUN_DIR" > "$ROOT/artifacts/latest-run.txt"
echo "artifact=$ROOT/artifacts/$(basename "$RUN_DIR").tar.gz"
