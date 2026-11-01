"""Fix the opt-in flag check in the vLLM 0.19.1 draft patch: skip the Qwen3.5 MTP rewrite only when
VLLM_QWEN35_DRAFT_NO_MTP_REWRITE == "1" (previously any non-empty value, including "0", disabled MTP)."""
PATH = "/home/euiyul/venv_vllm_018/lib/python3.10/site-packages/vllm/config/speculative.py"
src = open(PATH, encoding="utf-8").read()
old = 'not os.environ.get("VLLM_QWEN35_DRAFT_NO_MTP_REWRITE"):'
new = 'os.environ.get("VLLM_QWEN35_DRAFT_NO_MTP_REWRITE") != "1":'
if old in src:
    src = src.replace(old, new)
    open(PATH, "w", encoding="utf-8").write(src)
    print("flag check fixed")
else:
    print("pattern not found; current:", new in src)
