#!/usr/bin/env bash
# Exactness check of the draft-model path (Qwen3.5-0.8B draft, Qwen3.5-2B zero-shot target).
cd "$(dirname "$0")"
P=/home/euiyul/venv_vllm_018/bin/python3
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
D=/home/euiyul/.claude/jobs/81b8ceb3/tmp/draft_nomrope
$P verify_draft_exact.py --method base --target "$Z" --out logs_v3/exact_base.json > logs_v3/exact_base.log 2>&1
VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 VLLM_DRAFT_RELAX_KV_GROUP=1 $P verify_draft_exact.py --method draft --target "$Z" --draft "$D" --out logs_v3/exact_draft.json > logs_v3/exact_draft.log 2>&1
echo EXACT_DONE > logs_v3/exact.done
