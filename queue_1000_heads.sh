#!/usr/bin/env bash
# 1000-sample setting, corrected: heads are extracted from the 1000 training prompts
# (earlier extraction silently used the 256-prompt file), then the 7-method sweep runs.
cd "$(dirname "$0")"
PY=/home/euiyul/venv_vllm_018/bin/python3
TR=/home/euiyul/venv_train/bin/python3
M=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
L1=$M/lora1000_full
F1=$M/full1000_full
mkdir -p logs_v4

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  $TR build_heads.py extract --model "$T" --out $H/$tn --train_file train1000.jsonl > logs_v4/extract_$tn.log 2>&1
  echo "extract $tn exit=$?"
  $TR build_heads.py medusa --feats $H/$tn/feats.pt --out $H/$tn --max_pos 64 --epochs 2 > logs_v4/medusa_$tn.log 2>&1
  echo "medusa $tn exit=$?"
  $TR build_heads.py eagle3 --feats $H/$tn/feats.pt --out $H/$tn --epochs 2 > logs_v4/eagle3_$tn.log 2>&1
  echo "eagle3 $tn exit=$?"
done

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  for m in base ngram ngram_gpu suffix mtp medusa eagle3; do
    extra=""
    if [ "$m" = medusa ] || [ "$m" = eagle3 ]; then extra="--head $H/$tn/$m"; fi
    $PY vllm_squad_bench.py --method $m --target "$T" $extra --n 200 --out results_v4_squad.jsonl > logs_v4/squad_${tn}_${m}.log 2>&1
    grep "^RESULT" logs_v4/squad_${tn}_${m}.log | tail -n 1
    $PY gsm8k_vllm_bench.py --method $m --target "$T" $extra --out results_v4_gsm8k.jsonl > logs_v4/gsm8k_${tn}_${m}.log 2>&1
    grep "^RESULT" logs_v4/gsm8k_${tn}_${m}.log | tail -n 1
  done
done
echo DONE_1000_FINISH
