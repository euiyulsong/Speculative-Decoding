"""Copy the MTP head weights (mtp.*) from the base checkpoint into a merged checkpoint.

HF save of the merged target usually drops the MTP tensors (the HF model class has no MTP head),
so vLLM's native MTP method would have no weights. This writes mtp.safetensors next to the merged
weights and registers the keys in model.safetensors.index.json.

Usage: python3 add_mtp.py <base_snapshot_dir> <merged_dir>
"""
import glob
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

base_dir, merged_dir = sys.argv[1], sys.argv[2]

mtp = {}
for f in glob.glob(os.path.join(base_dir, "*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("mtp."):
                mtp[k] = sf.get_tensor(k)
print("mtp tensors in base:", len(mtp))
if not mtp:
    raise SystemExit("no mtp.* tensors found in base checkpoint")

out_name = "mtp.safetensors"
save_file(mtp, os.path.join(merged_dir, out_name), metadata={"format": "pt"})

idx_path = os.path.join(merged_dir, "model.safetensors.index.json")
if os.path.exists(idx_path):
    idx = json.load(open(idx_path, encoding="utf-8"))
    for k in mtp:
        idx["weight_map"][k] = out_name
    json.dump(idx, open(idx_path, "w", encoding="utf-8"), indent=2)
    print("index updated")
else:
    print("single-file checkpoint: mtp.safetensors added alongside model.safetensors")
print("done")
