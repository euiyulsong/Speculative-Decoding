"""Minimal vLLM 0.19.1 patch: let Qwen3.5's inner model expose EAGLE3 auxiliary hidden states.

Only the class declaration changes: Qwen3_5Model additionally inherits EagleModelMixin, which
vLLM's set_aux_hidden_state_layers() asserts on. The forward already collects aux states
(inherited from Qwen3NextModel). The original file is kept as qwen3_5.py.orig.
Applied to: /home/euiyul/venv_vllm_018 (CUDA 12 vLLM 0.19.1 install). Reverse with the .orig copy.
"""
import os
import re

PATH = ("/home/euiyul/venv_vllm_018/lib/python3.10/site-packages/vllm/"
        "model_executor/models/qwen3_5.py")
ORIG = PATH + ".orig"

src = open(PATH, encoding="utf-8").read()
if "EagleModelMixin" in src.split("class Qwen3_5Model")[1][:200]:
    print("already patched")
    raise SystemExit(0)
if not os.path.exists(ORIG):
    open(ORIG, "w", encoding="utf-8").write(src)

new, n = re.subn(r"class Qwen3_5Model\(Qwen3NextModel\):",
                 "class Qwen3_5Model(Qwen3NextModel, EagleModelMixin):", src)
assert n == 1, "pattern not found"
# make sure EagleModelMixin is importable in this module
if "EagleModelMixin" not in src.split("class Qwen3_5Model")[0]:
    new = new.replace("from .interfaces import (", "from .interfaces import (\n    EagleModelMixin,", 1)
    assert "EagleModelMixin," in new, "import line not found"
open(PATH, "w", encoding="utf-8").write(new)
print("patched; backup at", ORIG)
