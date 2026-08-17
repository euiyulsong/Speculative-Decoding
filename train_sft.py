import argparse, json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

MAX_LEN = 768
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class TrainDataset(Dataset):
    def __init__(self):
        with open("train.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(x) for x in f]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        x = self.rows[idx]
        prompt = x["prompt"]
        full = prompt + " " + x["answer"] + tokenizer.eos_token
        enc = tokenizer(full, max_length=MAX_LEN, truncation=True, padding="max_length")
        pids = tokenizer(prompt, max_length=MAX_LEN, truncation=True)["input_ids"]
        labels = enc["input_ids"].copy()
        for i in range(min(len(pids), len(labels))):
            labels[i] = -100
        for i, t in enumerate(enc["input_ids"]):
            if t == tokenizer.pad_token_id:
                labels[i] = -100
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

targs = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(model=model, args=targs, train_dataset=TrainDataset())
trainer.train()
trainer.save_model(args.output)
tokenizer.save_pretrained(args.output)
