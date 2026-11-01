import glob
import json
import struct


def keys(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return set(k for k in h if k != "__metadata__")


base = set()
for f in glob.glob("/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/*.safetensors"):
    base |= keys(f)
lora = keys("/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora2b_merged/model.safetensors")
print("base keys", len(base), "lora keys", len(lora))
print("only in lora:", sorted(lora - base)[:3])
print("only in base:", sorted(base - lora)[:3])
c = json.load(open("/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora2b_merged/config.json"))
print("lora model_type", c.get("model_type"), "| text", c.get("text_config", {}).get("model_type"))
