"""Minimal vLLM 0.19.1 patch: allow a Qwen3.5 model to be used as a draft model.

vLLM's SpeculativeConfig.hf_config_override rewrites any qwen3_5 draft config to the MTP loader
(qwen3_5_mtp), which fails for a full Qwen3.5-0.8B draft (hidden 1024 vs target 2048). This adds an
opt-in guard: when VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 is set, the rewrite is skipped. The override
only applies to draft configs, so the target and the native MTP method are unaffected.
Original file kept as speculative.py.orig.
"""
import os
import re

PATH = ("/home/euiyul/venv_vllm_018/lib/python3.10/site-packages/vllm/config/speculative.py")
ORIG = PATH + ".orig"

src = open(PATH, encoding="utf-8").read()
if "VLLM_QWEN35_DRAFT_NO_MTP_REWRITE" in src:
    print("already patched")
    raise SystemExit(0)
if not os.path.exists(ORIG):
    open(ORIG, "w", encoding="utf-8").write(src)

old = 'if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):'
assert src.count(old) == 1, src.count(old)
new = ('if hf_config.model_type in ("qwen3_5", "qwen3_5_moe") and not '
       'os.environ.get("VLLM_QWEN35_DRAFT_NO_MTP_REWRITE"):')
src = src.replace(old, new)
if not re.search(r"^import os", src, re.M):
    src = "import os\n" + src
open(PATH, "w", encoding="utf-8").write(src)
print("patched; backup at", ORIG)
