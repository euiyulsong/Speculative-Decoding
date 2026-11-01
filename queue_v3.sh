#!/usr/bin/env bash
# Full sweep on vLLM 0.19.1 (one engine at a time).
# targets: zero-shot (1-shot prompt, untrained), LoRA (SFT, rank 32), full SFT (all parameters)
# methods: base, ngram, ngram_gpu, suffix, mtp (all targets); medusa, eagle3 (targets with heads)
# tasks: SQuAD 1-shot (200), GSM8K 1-shot (100)
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
L=$M/lora2b_full
F=$M/full2b_full
mkdir -p logs_v3/q
export VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=0

run() {  # run <target_name> <target_path> <method> [head]
  local tname=$1 tpath=$2 m=$3 head=${4:-}
  local tag="${tname}_${m}"
  local extra=""
  if [ -n "$head" ]; then extra="--head $head"; fi
  echo "[$tag] squad"
  $PY vllm_squad_bench.py --method "$m" --target "$tpath" $extra --n 200 --out results_v3_squad.jsonl \
    > "logs_v3/q/squad_${tag}.log" 2>&1 || echo "[$tag] squad FAILED"
  grep "^RESULT" "logs_v3/q/squad_${tag}.log" | tail -n 1
  echo "[$tag] gsm8k"
  $PY gsm8k_vllm_bench.py --method "$m" --target "$tpath" $extra --out results_v3_gsm8k.jsonl \
    > "logs_v3/q/gsm8k_${tag}.log" 2>&1 || echo "[$tag] gsm8k FAILED"
  grep "^RESULT" "logs_v3/q/gsm8k_${tag}.log" | tail -n 1
}

for m in base ngram ngram_gpu suffix mtp; do run zero "$Z" "$m"; done
for m in medusa eagle3; do run zero "$Z" "$m" "$H/zero/$m"; done
for m in base ngram ngram_gpu suffix mtp; do run lora "$L" "$m"; done
for m in medusa eagle3; do run lora "$L" "$m" "$H/lora/$m"; done
for m in base ngram ngram_gpu suffix mtp; do run full "$F" "$m"; done
echo DONE
