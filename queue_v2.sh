#!/usr/bin/env bash
# Queue for the v2 sweep (vLLM 0.19.1): targets {zero-shot base, LoRA-trained} x methods
# {base, ngram, ngram_gpu, suffix, mtp} x tasks {SQuAD 1-shot 200, GSM8K 1-shot 100}.
# Runs strictly one engine at a time (GPU memory).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
BASE_SNAP=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
LORA=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora_merged
mkdir -p logs_v2

# wait for LoRA training + MTP copy
until grep -q "^SAVED" logs_v2/lora_train.log 2>/dev/null; do sleep 30; done
$PY add_mtp.py "$BASE_SNAP" "$LORA" > logs_v2/add_mtp.log 2>&1 || echo "add_mtp FAILED"

for target_name in zero_shot lora; do
  if [ "$target_name" = zero_shot ]; then TARGET=Qwen/Qwen3.5-0.8B; else TARGET="$LORA"; fi
  for m in base ngram ngram_gpu suffix mtp; do
    tag="${target_name}_${m}"
    echo "[$tag] squad"
    $PY vllm_squad_bench.py --method "$m" --target "$TARGET" --n 200 --out results_v2_squad.jsonl \
      > "logs_v2/squad_${tag}.log" 2>&1 || echo "[$tag] squad FAILED"
    grep "^RESULT" "logs_v2/squad_${tag}.log" | tail -n 1
    echo "[$tag] gsm8k"
    $PY gsm8k_vllm_bench.py --method "$m" --target "$TARGET" --out results_v2_gsm8k.jsonl \
      > "logs_v2/gsm8k_${tag}.log" 2>&1 || echo "[$tag] gsm8k FAILED"
    grep "^RESULT" "logs_v2/gsm8k_${tag}.log" | tail -n 1
  done
done
echo DONE
