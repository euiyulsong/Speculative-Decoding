"""Full-parameter SFT of the target on the 256 SQuAD prompts (the repo's original train_sft.py method).

Same data as train_lora.py (prompt + answer + EOS, loss on answer tokens only), but all parameters are
trained. To fit a 24 GB GPU the weights, gradients and AdamW states are kept in bf16 and gradient
checkpointing is on. The result is saved as a text-only HF checkpoint; overlay_full.py then writes a
complete checkpoint with the base config and vision tower.

Usage: python3 train_full.py --model Qwen/Qwen3.5-2B --out <dir>
"""
import argparse
import json
import math
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LEN = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_file", default="train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(x) for x in open(args.train_file, encoding="utf-8")]

    def encode(row):
        prompt = row["prompt"]
        full = prompt + " " + row["answer"] + tok.eos_token
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"]
        p_len = len(tok(prompt, add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"])
        labels = [-100] * min(p_len, len(ids)) + ids[min(p_len, len(ids)):]
        return torch.tensor([ids]), torch.tensor([labels])

    data = [encode(r) for r in rows]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(data) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / max(1, total_steps)))

    step, losses = 0, []
    for ep in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for i, idx in enumerate(order):
            x, y = data[idx]
            out = model(input_ids=x.cuda(), labels=y.cuda())
            (out.loss / args.accum).backward()
            losses.append(out.loss.item())
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"ep {ep} step {step}/{total_steps} loss(last80)={sum(losses[-80:])/len(losses[-80:]):.4f}", flush=True)
        if len(order) % args.accum:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
