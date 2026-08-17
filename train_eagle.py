import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["teacher", "onpolicy"], required=True)
args = parser.parse_args()

TARGET = "./target_sft"
MAX_LEN = 768
EPOCHS = 3
LR = 1e-4
DEVICE = "cuda"
OUT = "eagle_teacher.pt" if args.mode == "teacher" else "eagle_onpolicy.pt"

tokenizer = AutoTokenizer.from_pretrained(TARGET)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to(DEVICE)
target.eval()
for p in target.parameters():
    p.requires_grad = False

H = target.config.hidden_size
V = target.config.vocab_size
N_LAYERS = target.config.num_hidden_layers
LAYER_IDS = [2, N_LAYERS // 2, N_LAYERS - 2]

class DS(Dataset):
    def __init__(self):
        with open("train.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(x) for x in f]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]

@torch.inference_mode()
def make_sequence(row):
    prompt = row["prompt"]
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN - 32,
    )
    prompt_len = prompt_enc.input_ids.shape[1]

    if args.mode == "teacher":
        full = prompt + " " + row["answer"] + tokenizer.eos_token
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        return enc.input_ids[0], prompt_len

    inputs = {k: v.to(DEVICE) for k, v in prompt_enc.items()}
    generated = target.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return generated[0].cpu(), prompt_len

class Eagle3(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.input_proj = nn.Linear(hidden_size * 2, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, hidden_states):
        fused = self.feature_proj(torch.cat(hidden_states, dim=-1))
        tok = self.token_embed(input_ids)
        x = self.input_proj(torch.cat([tok, fused], dim=-1))
        T = x.shape[1]
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        x = self.decoder(x, mask=causal)
        return self.lm_head(self.norm(x))

eagle = Eagle3(H, V).to(DEVICE, dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(eagle.parameters(), lr=LR)
rows = DS()

for epoch in range(EPOCHS):
    eagle.train()
    indices = torch.randperm(len(rows)).tolist()

    for step, idx in enumerate(indices):
        row = rows[idx]
        input_ids, prompt_len = make_sequence(row)
        input_ids = input_ids.unsqueeze(0).to(DEVICE)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            target_out = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        selected = [target_out.hidden_states[i] for i in LAYER_IDS]
        logits = eagle(
            input_ids[:, :-1],
            [h[:, :-1] for h in selected],
        )
        labels = input_ids[:, 1:]
        mask = torch.ones_like(labels)
        cutoff = max(prompt_len - 1, 0)
        mask[:, :cutoff] = 0

        loss_tokens = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

        loss = (loss_tokens * mask).sum() / mask.sum().clamp_min(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eagle.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(
                f"eagle mode={args.mode} epoch={epoch} "
                f"step={step} loss={loss.item():.4f}"
            )

torch.save({
    "model": eagle.state_dict(),
    "hidden_size": H,
    "vocab_size": V,
    "layer_ids": LAYER_IDS,
    "target": TARGET,
}, OUT)

print("saved", OUT)
