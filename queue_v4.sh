#!/usr/bin/env bash
# Chain after queue_v3b: (A) draft-model exactness + sweep on 256 targets, (B) 1000-sample setting:
# train LoRA and full SFT on 1000 SQuAD prompts, heads, and the full method sweep.
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

# wait for the earlier queue (Medusa/EAGLE3 for the full 256 target, their runs)
until grep -q DONE_B queue_v3b.out 2>/dev/null; do sleep 60; done

echo "[draft exact] start"
bash run_draft_exact.sh
DENV="VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 VLLM_DRAFT_RELAX_KV_GROUP=1"  # draft runs only (MTP needs the rewrite)
echo "[draft exact] result"
$PY -c "import json;b=json.load(open('logs_v3/exact_base.json'));d=json.load(open('logs_v3/exact_draft.json'));print('identical', sum(x==y for x,y in zip(b,d)),'/',len(b))" 2>&1 | tail -n 1

# draft sweep on the 256-setting targets (draft = Qwen3.5-0.8B without M-RoPE keys)
for tn in zero lora full; do
  case $tn in zero) T=$Z;; lora) T=$L;; full) T=$F;; esac
  echo "[${tn}_draft] squad"
  env $DENV $PY vllm_squad_bench.py --method draft --target "$T" --draft "$D" --n 200 --out results_v3_squad.jsonl > logs_v4/squad_${tn}_draft.log 2>&1 || echo "FAILED squad ${tn}_draft"
  grep "^RESULT" logs_v4/squad_${tn}_draft.log | tail -n 1
  echo "[${tn}_draft] gsm8k"
  env $DENV $PY gsm8k_vllm_bench.py --method draft --target "$T" --draft "$D" --out results_v3_gsm8k.jsonl > logs_v4/gsm8k_${tn}_draft.log 2>&1 || echo "FAILED gsm8k ${tn}_draft"
  grep "^RESULT" logs_v4/gsm8k_${tn}_draft.log | tail -n 1
done

# (B) 1000-sample setting
$TR train_lora.py --model Qwen/Qwen3.5-2B --train_file train1000.jsonl --out $M/lora1000_raw > logs_v4/lora1000_train.log 2>&1
$PY add_mtp.py /home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc $M/lora1000_raw > /dev/null 2>&1
$TR overlay_ckpt.py $M/lora1000_raw $L1 > /dev/null 2>&1
$TR train_full.py --model Qwen/Qwen3.5-2B --train_file train1000.jsonl --out $M/full1000_raw > logs_v4/full1000_train.log 2>&1
$TR overlay_ckpt.py $M/full1000_raw $F1 > /dev/null 2>&1

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  $TR build_heads.py extract --model "$T" --out $H/$tn > logs_v4/extract_$tn.log 2>&1
  $TR build_heads.py medusa --feats $H/$tn/feats.pt --out $H/$tn --max_pos 64 --epochs 2 > logs_v4/medusa_$tn.log 2>&1
  $TR build_heads.py eagle3 --feats $H/$tn/feats.pt --out $H/$tn --epochs 2 > logs_v4/eagle3_$tn.log 2>&1
done

for tn in lora1000 full1000; do
  case $tn in lora1000) T=$L1;; full1000) T=$F1;; esac
  for m in base ngram ngram_gpu suffix mtp draft medusa eagle3; do
    extra=""; dflag=""
    if [ "$m" = medusa ] || [ "$m" = eagle3 ]; then extra="--head $H/$tn/$m"; fi
    if [ "$m" = draft ]; then dflag="--draft $D"; fi
    ev=""; if [ "$m" = draft ]; then ev="$DENV"; fi
    echo "[${tn}_${m}] squad"
    env $ev $PY vllm_squad_bench.py --method $m --target "$T" $extra $dflag --n 200 --out results_v4_squad.jsonl > logs_v4/squad_${tn}_${m}.log 2>&1 || echo "FAILED squad ${tn}_${m}"
    grep "^RESULT" logs_v4/squad_${tn}_${m}.log | tail -n 1
    echo "[${tn}_${m}] gsm8k"
    env $ev $PY gsm8k_vllm_bench.py --method $m --target "$T" $extra $dflag --out results_v4_gsm8k.jsonl > logs_v4/gsm8k_${tn}_${m}.log 2>&1 || echo "FAILED gsm8k ${tn}_${m}"
    grep "^RESULT" logs_v4/gsm8k_${tn}_${m}.log | tail -n 1
  done
done
echo DONE_V4
