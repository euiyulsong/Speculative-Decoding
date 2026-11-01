#!/usr/bin/env bash
# Finish the 1000-sample setting: overlay the trained checkpoints (path fixed), extract features and
# train Medusa/EAGLE3 heads, then measure 7 methods x 2 tasks (draft is blocked on vLLM 0.19.1).
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
TR=/home/euiyul/venv_train/bin/python3
OV=/home/euiyul/.claude/jobs/81b8ceb3/tmp/overlay_ckpt.py
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
L1=$M/lora1000_full
F1=$M/full1000_full
mkdir -p logs_v4

$TR $OV $M/lora1000_raw $L1 > logs_v4/lora1000_overlay.log 2>&1
$TR $OV $M/full1000_raw $F1 > logs_v4/full1000_overlay.log 2>&1

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  $TR build_heads.py extract --model "$T" --out $H/$tn > logs_v4/extract_$tn.log 2>&1
  $TR build_heads.py medusa --feats $H/$tn/feats.pt --out $H/$tn --max_pos 64 --epochs 2 > logs_v4/medusa_$tn.log 2>&1
  $TR build_heads.py eagle3 --feats $H/$tn/feats.pt --out $H/$tn --epochs 2 > logs_v4/eagle3_$tn.log 2>&1
done

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  for m in base ngram ngram_gpu suffix mtp medusa eagle3; do
    extra=""
    if [ "$m" = medusa ] || [ "$m" = eagle3 ]; then extra="--head $H/$tn"; fi
    if [ "$m" = medusa ] || [ "$m" = eagle3 ]; then extra="--head $H/$tn/$m"; fi
    $PY vllm_squad_bench.py --method $m --target "$T" $extra --n 200 --out results_v4_squad.jsonl > logs_v4/squad_${tn}_${m}.log 2>&1
    grep "^RESULT" logs_v4/squad_${tn}_${m}.log | tail -n 1
    $PY gsm8k_vllm_bench.py --method $m --target "$T" $extra --out results_v4_gsm8k.jsonl > logs_v4/gsm8k_${tn}_${m}.log 2>&1
    grep "^RESULT" logs_v4/gsm8k_${tn}_${m}.log | tail -n 1
  done
done
echo DONE_1000_FINISH
