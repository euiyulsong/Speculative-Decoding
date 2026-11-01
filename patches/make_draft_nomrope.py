"""Make a Qwen3.5-0.8B draft folder without M-RoPE config keys.

For text-only inputs the three M-RoPE position streams are identical, so plain RoPE with the same
theta and partial_rotary_factor computes the same thing. vLLM 0.19.1 refuses M-RoPE for draft models,
so the mrope_* keys are removed from rope_parameters. Weights are symlinked, architecture unchanged.
"""
import json
import os

SNAP = "/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
OUT = "/home/euiyul/.claude/jobs/81b8ceb3/tmp/draft_nomrope"
os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(os.path.join(SNAP, "config.json"), encoding="utf-8"))


def strip_mrope(obj):
    if isinstance(obj, dict):
        if "rope_parameters" in obj and isinstance(obj["rope_parameters"], dict):
            rp = obj["rope_parameters"]
            for k in ["mrope_interleaved", "mrope_section"]:
                rp.pop(k, None)
        for v in obj.values():
            strip_mrope(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_mrope(v)


strip_mrope(cfg)
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w", encoding="utf-8"), indent=1)

for name in os.listdir(SNAP):
    if name == "config.json":
        continue
    dst = os.path.join(OUT, name)
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(os.path.join(SNAP, name)), dst)
print("draft folder ready:", OUT)
