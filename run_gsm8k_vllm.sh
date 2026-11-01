#!/usr/bin/env bash
# GSM8K 1-shot, first 100 test items, vLLM native speculative decoding (one engine per method).
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
mkdir -p logs_vllm_gsm8k
for m in base ngram mtp; do
  echo "[$m] start"
  $PY gsm8k_vllm_bench.py --method "$m" > "logs_vllm_gsm8k/$m.log" 2>&1 || echo "[$m] FAILED"
  grep "^RESULT" "logs_vllm_gsm8k/$m.log" | tail -n 1
done
echo DONE
