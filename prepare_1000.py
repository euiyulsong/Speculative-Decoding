"""Write train1000.jsonl: 1000 SQuAD training prompts (1-shot format, same demo as prepare.py).
The evaluation file (eval.jsonl, 200 validation items) is unchanged."""
import json

from datasets import load_dataset

from common import make_prompt

N_TRAIN = 1000
ds = load_dataset("rajpurkar/squad")
train = ds["train"]
demo = train[0]
with open("train1000.jsonl", "w", encoding="utf-8") as f:
    for i in range(1, 1 + N_TRAIN):
        x = train[i]
        row = {"prompt": make_prompt(x, demo), "answer": x["answers"]["text"][0]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("prepared", N_TRAIN)
