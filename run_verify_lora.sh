#!/usr/bin/env bash
# Re-verify LoRA-target Medusa with the full (base-config) LoRA checkpoint.
cd "$(dirname "$0")"
P=/home/euiyul/venv_vllm_018/bin/python3
H=/home/euiyul/.claude/jobs/81b8ceb3/tmp/heads
L=/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora2b_full
$P verify_heads.py --target "$L" --method medusa --head "$H/lora/medusa" > logs_v3/verify_lora_medusa.log 2>&1
echo LORA_MEDUSA_DONE > logs_v3/verify_lora.done
