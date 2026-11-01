#!/usr/bin/env bash
# Exactness check, draft path only (literal env, no indirection), then compare with the base outputs.
cd "$(dirname "$0")"
Z=/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
D=/home/euiyul/.claude/jobs/81b8ceb3/tmp/draft_nomrope
export VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1
export VLLM_DRAFT_RELAX_KV_GROUP=1
/home/euiyul/venv_vllm_018/bin/python3 verify_draft_exact.py --method draft --target "$Z" --draft "$D" --out logs_v3/exact_draft.json > logs_v3/exact_draft.log 2>&1
echo "exit=$?"
/home/euiyul/venv_vllm_018/bin/python3 -c "import json;b=json.load(open('logs_v3/exact_base.json'));d=json.load(open('logs_v3/exact_draft.json'));print('identical', sum(x==y for x,y in zip(b,d)),'/',len(b))" 2>&1 | tail -n 1
