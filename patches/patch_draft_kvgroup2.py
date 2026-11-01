"""Second step of the opt-in draft patch: when relaxing the single-group check, pick a KV group whose
spec is an attention spec (has dtype), not a Mamba/linear-attention state group."""
PATH = "/home/euiyul/venv_vllm_018/lib/python3.10/site-packages/vllm/v1/spec_decode/eagle.py"
src = open(PATH, encoding="utf-8").read()
old = """        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break
"""
new = """        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                if os.environ.get("VLLM_DRAFT_RELAX_KV_GROUP") == "1" and not hasattr(group.kv_cache_spec, "dtype"):
                    continue  # opt-in: skip Mamba/linear-attention state groups
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break
"""
if "skip Mamba/linear-attention state groups" in src:
    print("already applied")
else:
    assert src.count(old) == 1, src.count(old)
    src = src.replace(old, new)
    open(PATH, "w", encoding="utf-8").write(src)
    print("applied")
