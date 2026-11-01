#!/usr/bin/env bash
# Follow-up for queue_v3: (1) re-run zero-shot MTP after the patch-flag fix,
# (2) extract hidden states of the full-SFT target, train its Medusa and EAGLE3 heads,
# (3) run those two methods on SQuAD and GSM8K.
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
TR=/home/euiyul/venv_train/bin/python3
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
F=$M/full2b_full
mkdir -p logs_v3/q logs_v3/heads

echo "[zero_mtp rerun] squad"
$PY vllm_squad_bench.py --method mtp --target "$Z" --n 200 --out results_v3_squad.jsonl > logs_v3/q/squad_zero_mtp.log 2>&1 || echo "zero_mtp squad FAILED"
grep "^RESULT" logs_v3/q/squad_zero_mtp.log | tail -n 1
echo "[zero_mtp rerun] gsm8k"
$PY gsm8k_vllm_bench.py --method mtp --target "$Z" --out results_v3_gsm8k.jsonl > logs_v3/q/gsm8k_zero_mtp.log 2>&1 || echo "zero_mtp gsm8k FAILED"
grep "^RESULT" logs_v3/q/gsm8k_zero_mtp.log | tail -n 1

echo "[full heads] extract"
mkdir -p $H/full
$TR build_heads.py extract --model "$F" --out $H/full > logs_v3/heads/extract_full.log 2>&1 || echo "extract FAILED"
echo "[full heads] medusa"
$TR build_heads.py medusa --feats $H/full/feats.pt --out $H/full > logs_v3/heads/medusa_full.log 2>&1 || echo "medusa FAILED"
echo "[full heads] eagle3"
$TR build_heads.py eagle3 --feats $H/full/feats.pt --out $H/full > logs_v3/heads/eagle3_full.log 2>&1 || echo "eagle3 FAILED"

for m in medusa eagle3; do
  if [ -f "$H/full/$m/model.safetensors" ]; then
    echo "[full_$m] squad"
    $PY vllm_squad_bench.py --method "$m" --target "$F" --head "$H/full/$m" --n 200 --out results_v3_squad.jsonl > "logs_v3/q/squad_full_$m.log" 2>&1 || echo "full_$m squad FAILED"
    grep "^RESULT" "logs_v3/q/squad_full_$m.log" | tail -n 1
    echo "[full_$m] gsm8k"
    $PY gsm8k_vllm_bench.py --method "$m" --target "$F" --head "$H/full/$m" --out results_v3_gsm8k.jsonl > "logs_v3/q/gsm8k_full_$m.log" 2>&1 || echo "full_$m gsm8k FAILED"
    grep "^RESULT" "logs_v3/q/gsm8k_full_$m.log" | tail -n 1
  fi
done
echo DONE_B
