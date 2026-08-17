#!/usr/bin/env bash
set -euo pipefail

# Official vLLM Speculators EAGLE-3 training, fair off-policy vs on-policy.
# Existing custom train_eagle.py is intentionally left untouched.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPECULATORS_DIR="${SPECULATORS_DIR:-$HOME/speculators}"
TARGET="${TARGET:-./target_sft}"
PORT="${PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
SEQ_LEN="${SEQ_LEN:-768}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
TTT_STEPS="${TTT_STEPS:-3}"
CONCURRENCY="${CONCURRENCY:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SEED="${SEED:-42}"

if [[ ! -f "$SPECULATORS_DIR/scripts/train.py" ]]; then
  echo "ERROR: SPECULATORS_DIR=$SPECULATORS_DIR does not look like vllm-project/speculators"
  echo "Clone/install speculators and set SPECULATORS_DIR=/path/to/speculators"
  exit 2
fi
if [[ ! -d "$TARGET" ]]; then
  echo "ERROR: target checkpoint not found: $TARGET"
  echo "Run: python3 train_sft.py --model Qwen/Qwen3.5-2B --output ./target_sft"
  exit 2
fi

mkdir -p official_eagle3/{data_raw,prepared_off,prepared_on,hidden_off,hidden_on,ckpt_off,ckpt_on,logs}

wait_for_server() {
  local url="http://127.0.0.1:${PORT}/v1/models"
  for _ in $(seq 1 180); do
    if python3 - "$url" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=2).read()
except Exception:
    raise SystemExit(1)
PY
    then return 0; fi
    sleep 2
  done
  echo "ERROR: vLLM hidden-state server did not become ready"
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

echo "[1/7] Build SAME-question off-policy and on-policy datasets"
python3 official_eagle3/make_data.py \
  --input train.jsonl \
  --target "$TARGET" \
  --output-dir official_eagle3/data_raw \
  --max-samples "$MAX_SAMPLES" \
  --seed "$SEED"

echo "[2/7] Official Speculators preprocessing"
python3 "$SPECULATORS_DIR/scripts/prepare_data.py" \
  --model "$TARGET" \
  --data official_eagle3/data_raw/offpolicy.jsonl \
  --output official_eagle3/prepared_off \
  --max-samples "$MAX_SAMPLES" \
  --seq-length "$SEQ_LEN" \
  --seed "$SEED" \
  --overwrite

python3 "$SPECULATORS_DIR/scripts/prepare_data.py" \
  --model "$TARGET" \
  --data official_eagle3/data_raw/onpolicy.jsonl \
  --output official_eagle3/prepared_on \
  --max-samples "$MAX_SAMPLES" \
  --seq-length "$SEQ_LEN" \
  --seed "$SEED" \
  --overwrite

echo "[3/7] Start verifier with official hidden-state extraction wrapper"
python3 "$SPECULATORS_DIR/scripts/launch_vllm.py" "$TARGET" -- \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  > official_eagle3/logs/hidden_server.log 2>&1 &
SERVER_PID=$!
wait_for_server

echo "[4/7] Cache OFF-policy verifier hidden states"
python3 "$SPECULATORS_DIR/scripts/data_generation_offline.py" \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --preprocessed-data official_eagle3/prepared_off \
  --output official_eagle3/hidden_off \
  --max-samples "$MAX_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --validate-outputs \
  --fail-on-error

echo "[5/7] Cache ON-policy verifier hidden states"
python3 "$SPECULATORS_DIR/scripts/data_generation_offline.py" \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --preprocessed-data official_eagle3/prepared_on \
  --output official_eagle3/hidden_on \
  --max-samples "$MAX_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --validate-outputs \
  --fail-on-error

stop_server
SERVER_PID=""

echo "[6/7] Train OFF-policy official EAGLE-3"
python3 "$SPECULATORS_DIR/scripts/train.py" \
  --speculator-type eagle3 \
  --verifier-name-or-path "$TARGET" \
  --data-path official_eagle3/prepared_off \
  --hidden-states-path official_eagle3/hidden_off \
  --on-missing raise \
  --save-path official_eagle3/ckpt_off \
  --num-layers 1 \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --total-seq-len "$SEQ_LEN" \
  --ttt-steps "$TTT_STEPS" \
  --optimizer adamw \
  --seed "$SEED" \
  --save-best \
  --no-resume-from-checkpoint

echo "[7/7] Train ON-policy official EAGLE-3"
python3 "$SPECULATORS_DIR/scripts/train.py" \
  --speculator-type eagle3 \
  --verifier-name-or-path "$TARGET" \
  --data-path official_eagle3/prepared_on \
  --hidden-states-path official_eagle3/hidden_on \
  --on-missing raise \
  --save-path official_eagle3/ckpt_on \
  --num-layers 1 \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --total-seq-len "$SEQ_LEN" \
  --ttt-steps "$TTT_STEPS" \
  --optimizer adamw \
  --seed "$SEED" \
  --save-best \
  --no-resume-from-checkpoint

echo
echo "DONE"
echo "OFF: official_eagle3/ckpt_off/checkpoint_best"
echo " ON: official_eagle3/ckpt_on/checkpoint_best"
echo "Next: python3 official_eagle3/benchmark_vllm.py"
