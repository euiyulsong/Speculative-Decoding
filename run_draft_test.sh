#!/usr/bin/env bash
# Draft-model verification: Qwen3.5-0.8B as draft for Qwen3.5-2B (zero-shot), patched vLLM 0.19.1.
cd "$(dirname "$0")"
P=/home/euiyul/venv_vllm_018/bin/python3
D=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 $P verify_draft.py --target "$Z" --draft "$D" --n 10 > logs_v3/verify_draft_zero.log 2>&1
echo DRAFT_TEST_DONE > logs_v3/verify_draft.done
