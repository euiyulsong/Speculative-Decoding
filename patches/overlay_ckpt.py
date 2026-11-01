"""Overlay trained language-model tensors onto the base Qwen3.5-2B checkpoint.

Usage: python3 overlay_ckpt.py <trained_dir> <out_dir>
Trained checkpoint keys must already be names present in the base checkpoint (checked by assert).
Keeps the base config, tokenizer, visual tower and MTP tensors.
"""
import glob
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/home/euiyul/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
trained, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

tensors = {}
for f in glob.glob(os.path.join(BASE, "*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)
n_base = len(tensors)
src_files = sorted(glob.glob(os.path.join(trained, "*.safetensors")))
n_new = 0
for f in src_files:
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k not in tensors:
                raise SystemExit(f"unexpected key {k}")
            tensors[k] = sf.get_tensor(k)
            n_new += 1
print("base tensors", n_base, "overlaid", n_new)
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
shutil.copy(os.path.join(BASE, "config.json"), os.path.join(out, "config.json"))
for name in os.listdir(BASE):
    if name.endswith((".json", ".txt", ".jinja")) and name != "config.json" and "index" not in name:
        shutil.copy(os.path.join(BASE, name), os.path.join(out, name))
print("done", out)
