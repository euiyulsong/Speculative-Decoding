#!/usr/bin/env bash
# Verify trained Medusa / EAGLE3 heads in vLLM 0.19.1 on zero-shot and LoRA Qwen3.5-2B targets.
cd "$(dirname "$0")"
P=/home/euiyul/venv_vllm_018/bin/python3
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
L=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora2b_merged
$P verify_heads.py --target "$Z" --method medusa --head "$H/zero/medusa" > logs_v3/verify_zero_medusa.log 2>&1
$P verify_heads.py --target "$Z" --method eagle3 --head "$H/zero/eagle3" > logs_v3/verify_zero_eagle3.log 2>&1
$P verify_heads.py --target "$L" --method medusa --head "$H/lora/medusa" > logs_v3/verify_lora_medusa.log 2>&1
$P verify_heads.py --target "$L" --method eagle3 --head "$H/lora/eagle3" > logs_v3/verify_lora_eagle3.log 2>&1
echo VERIFY_DONE > logs_v3/verify.done
