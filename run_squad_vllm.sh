#!/usr/bin/env bash
# SQuAD 1-shot, 200 eval items, vLLM native speculative decoding (one engine per method).
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
rm -f results_vllm_squad.jsonl
mkdir -p logs_vllm_squad
for m in ${METHODS:-base ngram draft mtp}; do
  echo "[$m] start"
  $PY vllm_squad_bench.py --method "$m" --n 200 > "logs_vllm_squad/$m.log" 2>&1 || echo "[$m] FAILED (see logs_vllm_squad/$m.log)"
  grep "^RESULT" "logs_vllm_squad/$m.log" | tail -n 1 || true
done
echo DONE
