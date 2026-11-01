#!/usr/bin/env bash
# 1000-sample setting: train LoRA and full SFT on 1000 SQuAD prompts, build Medusa/EAGLE3 heads,
# and run the method sweep (8 methods x 2 tasks). Starts after the draft queue.
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
TR=/home/euiyul/venv_train/bin/python3
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
D=/home/euiyul/.claude/jobs/81b8ceb3/tmp/draft_nomrope
BASE=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
L1=$M/lora1000_full
F1=$M/full1000_full
DENV="VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 VLLM_DRAFT_RELAX_KV_GROUP=1"
mkdir -p logs_v4

until grep -q DONE_DRAFT queue_draft.out 2>/dev/null; do sleep 60; done

$TR train_lora.py --model Qwen/Qwen3.5-2B --train_file train1000.jsonl --out $M/lora1000_raw > logs_v4/lora1000_train.log 2>&1
$TR overlay_ckpt.py $M/lora1000_raw $L1 > logs_v4/lora1000_overlay.log 2>&1
$TR train_full.py --model Qwen/Qwen3.5-2B --train_file train1000.jsonl --out $M/full1000_raw > logs_v4/full1000_train.log 2>&1
$TR overlay_ckpt.py $M/full1000_raw $F1 > logs_v4/full1000_overlay.log 2>&1

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  $TR build_heads.py extract --model "$T" --out $H/$tn > logs_v4/extract_$tn.log 2>&1
  $TR build_heads.py medusa --feats $H/$tn/feats.pt --out $H/$tn --max_pos 64 --epochs 2 > logs_v4/medusa_$tn.log 2>&1
  $TR build_heads.py eagle3 --feats $H/$tn/feats.pt --out $H/$tn --epochs 2 > logs_v4/eagle3_$tn.log 2>&1
done

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  for m in base ngram ngram_gpu suffix mtp draft medusa eagle3; do
    extra=""; dflag=""; ev=""
    if [ "$m" = medusa ] || [ "$m" = eagle3 ]; then extra="--head $H/$tn/$m"; fi
    if [ "$m" = draft ]; then dflag="--draft $D"; ev="$DENV"; fi
    echo "[${tn}_${m}] squad"
    env $ev $PY vllm_squad_bench.py --method $m --target "$T" $extra $dflag --n 200 --out results_v4_squad.jsonl > logs_v4/squad_${tn}_${m}.log 2>&1 || echo "FAILED squad ${tn}_${m}"
    grep "^RESULT" logs_v4/squad_${tn}_${m}.log | tail -n 1
    echo "[${tn}_${m}] gsm8k"
    env $ev $PY gsm8k_vllm_bench.py --method $m --target "$T" $extra $dflag --out results_v4_gsm8k.jsonl > logs_v4/gsm8k_${tn}_${m}.log 2>&1 || echo "FAILED gsm8k ${tn}_${m}"
    grep "^RESULT" logs_v4/gsm8k_${tn}_${m}.log | tail -n 1
  done
done
echo DONE_1000
