#!/usr/bin/env bash
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-2B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3.5-0.8B}"

echo "[1/7] prepare 1-shot SQuAD data"
python3 prepare.py

echo "[2/7] train target SFT"
python3 train_sft.py \
  --model "$TARGET_MODEL" \
  --output ./target_sft

echo "[3/7] train draft SFT"
python3 train_sft.py \
  --model "$DRAFT_MODEL" \
  --output ./draft_sft

echo "[4/7] train MTP"
python3 train_mtp.py

echo "[5/7] train EAGLE teacher-forcing"
python3 train_eagle.py --mode teacher

echo "[6/7] train EAGLE on-policy"
python3 train_eagle.py --mode onpolicy

echo "[7/7] benchmark all"
python3 benchmark_all.py

echo
echo "DONE"
echo "Results saved to results.json"
