#!/usr/bin/env bash
# After the running LoRA-1000 training: (1) overlay LoRA-1000, (2) draft exactness check,
# (3) draft sweep on the 256 targets (literal env), (4) full-SFT-1000 training, heads, method sweep.
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
TR=/home/euiyul/venv_train/bin/python3
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
D=/home/euiyul/.claude/jobs/81b8ceb3/tmp/draft_nomrope
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
L=$M/lora2b_full
F=$M/full2b_full
L1=$M/lora1000_full
F1=$M/full1000_full
mkdir -p logs_v4

while kill -0 476571 2>/dev/null; do sleep 30; done
$TR overlay_ckpt.py $M/lora1000_raw $L1 > logs_v4/lora1000_overlay.log 2>&1

# (2) draft exactness with literal environment
export VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1
export VLLM_DRAFT_RELAX_KV_GROUP=1
$PY verify_draft_exact.py --method draft --target "$Z" --draft "$D" --out logs_v3/exact_draft.json > logs_v3/exact_draft.log 2>&1
echo "exact exit=$?"
$PY -c "import json;b=json.load(open('logs_v3/exact_base.json'));d=json.load(open('logs_v3/exact_draft.json'));print('identical', sum(x==y for x,y in zip(b,d)),'/',len(b))" 2>&1 | tail -n 1

# (3) draft sweep on 256 targets
for tn in zero lora full; do
  case $tn in zero) T=$Z;; lora) T=$L;; full) T=$F;; esac
  $PY vllm_squad_bench.py --method draft --target "$T" --draft "$D" --n 200 --out results_v3_squad.jsonl > logs_v4/squad_${tn}_draft.log 2>&1 ; grep "^RESULT" logs_v4/squad_${tn}_draft.log | tail -n 1
  $PY gsm8k_vllm_bench.py --method draft --target "$T" --draft "$D" --out results_v3_gsm8k.jsonl > logs_v4/gsm8k_${tn}_draft.log 2>&1 ; grep "^RESULT" logs_v4/gsm8k_${tn}_draft.log | tail -n 1
done
unset VLLM_QWEN35_DRAFT_NO_MTP_REWRITE
unset VLLM_DRAFT_RELAX_KV_GROUP

# (4) full SFT on 1000, heads, method sweep
$TR train_full.py --model Qwen/Qwen3.5-2B --train_file train1000.jsonl --out $M/full1000_raw > logs_v4/full1000_train.log 2>&1
$TR overlay_ckpt.py $M/full1000_raw $F1 > logs_v4/full1000_overlay.log 2>&1
for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  $TR build_heads.py extract --model "$T" --out $H/$tn > logs_v4/extract_$tn.log 2>&1
  $TR build_heads.py medusa --feats $H/$tn/feats.pt --out $H/$tn --max_pos 64 --epochs 2 > logs_v4/medusa_$tn.log 2>&1
  $TR build_heads.py eagle3 --feats $H/$tn/feats.pt --out $H/$tn --epochs 2 > logs_v4/eagle3_$tn.log 2>&1
done
echo DONE_FIX
