import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET = "./target_sft"
OUT = "mtp_squad.pt"
MAX_LEN = 768
K = 4
EPOCHS = 3
LR = 1e-4
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(TARGET)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to(DEVICE)
target.eval()
for p in target.parameters():
    p.requires_grad = False

H = target.config.hidden_size
V = target.config.vocab_size

class DS(Dataset):
    def __init__(self):
        with open("train.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(x) for x in f]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]

def collate(batch):
    texts, answer_starts = [], []
    for x in batch:
        prompt = x["prompt"]
        text = prompt + " " + x["answer"] + tokenizer.eos_token
        texts.append(text)
        plen = len(tokenizer(prompt, truncation=True, max_length=MAX_LEN)["input_ids"])
        answer_starts.append(plen)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc, answer_starts

loader = DataLoader(DS(), batch_size=1, shuffle=True, collate_fn=collate)

class MTP(nn.Module):
    def __init__(self, hidden_size, vocab_size, k):
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(hidden_size)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(k)
        ])
    def forward(self, hidden):
        hidden = self.norm(hidden)
        return [head(hidden) for head in self.heads]

mtp = MTP(H, V, K).to(DEVICE, dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(mtp.parameters(), lr=LR)

for epoch in range(EPOCHS):
    mtp.train()
    for step, (enc, answer_starts) in enumerate(loader):
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            out = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = out.hidden_states[-1]
        logits_list = mtp(hidden)

        losses = []
        for k, logits in enumerate(logits_list, start=1):
            pred = logits[:, :-k]
            labels = input_ids[:, k:]
            mask = attention_mask[:, k:].clone()
            for b, start in enumerate(answer_starts):
                cutoff = max(start - k, 0)
                mask[b, :cutoff] = 0

            token_loss = F.cross_entropy(
                pred.reshape(-1, V),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            losses.append((token_loss * mask).sum() / mask.sum().clamp_min(1))

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(f"mtp epoch={epoch} step={step} loss={loss.item():.4f}")

torch.save({
    "model": mtp.state_dict(),
    "hidden_size": H,
    "vocab_size": V,
    "k": K,
    "target": TARGET,
}, OUT)
print("saved", OUT)
