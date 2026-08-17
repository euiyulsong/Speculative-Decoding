import json
from datasets import load_dataset
from common import make_prompt

N_TRAIN = 256
N_EVAL = 200

ds = load_dataset("rajpurkar/squad")
train = ds["train"]
val = ds["validation"]

demo = train[0]

with open("train.jsonl", "w", encoding="utf-8") as f:
    for i in range(1, 1 + N_TRAIN):
        x = train[i]
        row = {
            "prompt": make_prompt(x, demo),
            "answer": x["answers"]["text"][0],
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open("eval.jsonl", "w", encoding="utf-8") as f:
    for x in val.select(range(N_EVAL)):
        row = {
            "prompt": make_prompt(x, demo),
            "answers": x["answers"]["text"],
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("prepared:", N_TRAIN, "train,", N_EVAL, "eval, 1-shot")
