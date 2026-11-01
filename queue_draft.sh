#!/usr/bin/env bash
# Draft-model path on vLLM 0.19.1: exactness check, then sweep on the 256-setting targets.
# Draft runs need VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 (M-RoPE-free draft config) and
# VLLM_DRAFT_RELAX_KV_GROUP=1 (opt-in KV-group relaxation). Other runs do not get these variables.
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
D=/home/euiyul/.claude/jobs/81b8ceb3/tmp/draft_nomrope
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
L=$M/lora2b_full
F=$M/full2b_full
DENV="VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 VLLM_DRAFT_RELAX_KV_GROUP=1"
mkdir -p logs_v4

echo "[exact] base"
$PY verify_draft_exact.py --method base --target "$Z" --out logs_v3/exact_base.json > logs_v3/exact_base.log 2>&1
echo "[exact] draft"
env $DENV $PY verify_draft_exact.py --method draft --target "$Z" --draft "$D" --out logs_v3/exact_draft.json > logs_v3/exact_draft.log 2>&1 || echo "exact draft FAILED"
$PY -c "import json;b=json.load(open('logs_v3/exact_base.json'));d=json.load(open('logs_v3/exact_draft.json'));print('identical', sum(x==y for x,y in zip(b,d)),'/',len(b))" 2>&1 | tail -n 1

for tn in zero lora full; do
  case $tn in zero) T=$Z;; lora) T=$L;; full) T=$F;; esac
  echo "[${tn}_draft] squad"
  env $DENV $PY vllm_squad_bench.py --method draft --target "$T" --draft "$D" --n 200 --out results_v3_squad.jsonl > logs_v4/squad_${tn}_draft.log 2>&1 || echo "FAILED squad ${tn}_draft"
  grep "^RESULT" logs_v4/squad_${tn}_draft.log | tail -n 1
  echo "[${tn}_draft] gsm8k"
  env $DENV $PY gsm8k_vllm_bench.py --method draft --target "$T" --draft "$D" --out results_v3_gsm8k.jsonl > logs_v4/gsm8k_${tn}_draft.log 2>&1 || echo "FAILED gsm8k ${tn}_draft"
  grep "^RESULT" logs_v4/gsm8k_${tn}_draft.log | tail -n 1
done
echo DONE_DRAFT
