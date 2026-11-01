"""Opt-in vLLM 0.19.1 patch: allow draft layers spread over several KV-cache groups.

validate_same_kv_cache_group asserts that all drafting layers share one KV-cache group. For the
hybrid Qwen3.5-0.8B draft this fails (the draft has a different page size than the 2B target). With
VLLM_DRAFT_RELAX_KV_GROUP=1 the assertion becomes a warning and the first matching group is used.
Whether this is exact must be checked against the base model's greedy outputs (see verify_draft_exact.py).
Original file kept as eagle.py.orig.
"""
import os

PATH = "/home/euiyul/venv_vllm_018/lib/python3.10/site-packages/vllm/v1/spec_decode/eagle.py"
ORIG = PATH + ".orig"
src = open(PATH, encoding="utf-8").read()
if "VLLM_DRAFT_RELAX_KV_GROUP" in src:
    print("already patched")
    raise SystemExit(0)
if not os.path.exists(ORIG):
    open(ORIG, "w", encoding="utf-8").write(src)

old = """        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"
"""
new = """        _groups = set(
            [
                kv_cache_groups[layer_name]
                for layer_name in self._draft_attn_layer_names
            ]
        )
        if os.environ.get("VLLM_DRAFT_RELAX_KV_GROUP") == "1":
            import logging
            logging.getLogger(__name__).warning(
                "draft layers span KV groups %s; relaxed (opt-in)", sorted(_groups))
        else:
            assert len(_groups) == 1, (
                "All drafting layers should belong to the same kv cache group")
"""
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new)
if "\nimport os\n" not in src:
    src = "import os\n" + src
open(PATH, "w", encoding="utf-8").write(src)
print("patched eagle.py (opt-in VLLM_DRAFT_RELAX_KV_GROUP=1)")
