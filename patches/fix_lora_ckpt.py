"""Build a full Qwen3.5-2B checkpoint from the LoRA-merged text weights.

The merged checkpoint holds only the language-model tensors (320) with the same names as the base
checkpoint, and its config says model_type qwen3_5_text, which vLLM's bundled transformers 4.57
cannot read. This keeps the base config and visual tower, and overwrites the language tensors with
the LoRA-trained ones. The MTP tensors are copied from the base as well.
"""
import glob
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
LORA = "/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora2b_merged"
OUT = "/home/euiyul/.claude/jobs/81b8ceb3/tmp/models/lora2b_full"

os.makedirs(OUT, exist_ok=True)
tensors = {}
for f in glob.glob(os.path.join(BASE, "*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)
n_base = len(tensors)
with safe_open(os.path.join(LORA, "model.safetensors"), framework="pt") as sf:
    lora_keys = list(sf.keys())
    for k in lora_keys:
        assert k in tensors, f"unexpected key {k}"
        tensors[k] = sf.get_tensor(k)
print("base tensors", n_base, "replaced with LoRA", len(lora_keys))
save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
shutil.copy(os.path.join(BASE, "config.json"), os.path.join(OUT, "config.json"))
for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
             "video_preprocessor_config.json"]:
    src = os.path.join(BASE, name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, name))
print("done", OUT)
