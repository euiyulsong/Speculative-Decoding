"""Train Medusa and EAGLE3 heads for a Qwen3.5 target, in vLLM's checkpoint format.

Stages (run in venv_train, which has transformers with Qwen3.5 support):
  extract  : run the target on the SQuAD train prompts (prompt + answer) and cache
             last-layer hidden states (Medusa input) and layers (2, 12, 21) concatenated (EAGLE3 input)
  medusa   : K residual-block heads predicting tokens t+2 .. t+K+1 from h_t
  eagle3   : one llama-style decoder layer (vLLM Eagle3LlamaForCausalLM layout). Input at t is
             (token t+1 embedding, fc(aux_t)) and the label is token t+2 (teacher forcing, one step).
Both heads are saved as safetensors + config.json under <out>/medusa and <out>/eagle3.

Usage:
  python3 build_heads.py extract --model <target> --out <dir>
  python3 build_heads.py medusa  --feats <dir>/feats.pt --out <dir>
  python3 build_heads.py eagle3  --feats <dir>/feats.pt --out <dir>
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN = "train.jsonl"
AUX_LAYERS = (2, 12, 21)
K_MEDUSA = 4


def extract(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    rows = [json.loads(x) for x in open(args.train_file, encoding="utf-8")]
    data = []
    with torch.inference_mode():
        for r in rows:
            full = r["prompt"] + " " + r["answer"]
            ids = tok(full, add_special_tokens=False)["input_ids"][:768]
            plen = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            x = torch.tensor([ids], device="cuda")
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
            last = hs[-1][0].to("cpu")  # post-norm final hidden state (what lm_head reads)
            aux = torch.cat([hs[i][0] for i in AUX_LAYERS], dim=-1).to("cpu")
            data.append({"ids": torch.tensor(ids), "plen": plen, "last": last, "aux": aux})
    embed = model.get_input_embeddings().weight.detach().to("cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.save({"data": data, "embed": embed}, os.path.join(args.out, "feats.pt"))
    print("extracted", len(data), "sequences; hidden", tuple(data[0]["last"].shape), flush=True)


class ResidualBlock(nn.Module):
    def __init__(self, H, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(n_layers)])
        self.act = nn.SiLU()

    def forward(self, x):
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


def medusa(args):
    from safetensors.torch import save_file
    blob = torch.load(args.feats, weights_only=False)
    data, embed = blob["data"], blob["embed"]
    H, V = embed.shape[1], embed.shape[0]
    blocks = nn.ModuleList([ResidualBlock(H, 1) for _ in range(K_MEDUSA)]).cuda()
    heads = nn.ModuleList([nn.Linear(H, V, bias=False) for _ in range(K_MEDUSA)]).cuda()
    for h in heads:  # init each head from the (tied) target embedding / lm_head
        h.weight.data.copy_(embed.cuda().float().to(h.weight.dtype))
        h.weight.requires_grad_(True)
    for b in blocks:
        for l in b.layers:
            nn.init.zeros_(l.weight)
    params = list(blocks.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    random.seed(0)
    steps = 0
    for ep in range(args.epochs):
        random.shuffle(data)
        for d in data:
            h = d["last"].cuda().float()
            ids = d["ids"].cuda()
            plen = d["plen"]
            loss = 0.0
            for k in range(K_MEDUSA):
                # head k at position t predicts token t+2+k, for t >= plen-1 (answer region)
                T = h.shape[0]
                start, end = max(plen - 1, 0), T - 2 - k
                if end <= start:  # sequence too short for this head
                    continue
                t = torch.arange(start, end, device="cuda")
                if args.max_pos and t.numel() > args.max_pos:  # subsample positions for speed
                    t = t[torch.randperm(t.numel(), device="cuda")[: args.max_pos]]
                logits = heads[k](blocks[k](h[t]))
                loss = loss + F.cross_entropy(logits.float(), ids[t + 2 + k])
            if torch.is_tensor(loss):
                (loss / K_MEDUSA).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
        print(f"medusa epoch {ep} done, steps {steps}", flush=True)

    tensors = {}
    for i, b in enumerate(blocks):
        for j, l in enumerate(b.layers):
            tensors[f"blocks.{i}.layers.{j}.weight"] = l.weight.detach().to("cpu", torch.bfloat16).contiguous()
    for i, hd in enumerate(heads):
        tensors[f"lm_heads.{i}.weight"] = hd.weight.detach().to("cpu", torch.bfloat16).contiguous()
    out = os.path.join(args.out, "medusa")
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    cfg = {
        "architectures": ["MedusaModel"], "model_type": "medusa",
        "hidden_size": H, "vocab_size": V, "truncated_vocab_size": V,
        "num_heads": K_MEDUSA, "num_hidden_layers": 1,
        "original_lm_head": False, "medusa_fc_bias": False,
        "torch_dtype": "bfloat16",
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print("medusa saved", out, flush=True)


class Eagle3Layer(nn.Module):
    """Mirror of vLLM's Eagle3 llama layer (layer 0): attention over cat(embed, hidden)."""

    def __init__(self, H, I, nh, nkv, hd):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.input_layernorm = nn.RMSNorm(H, eps=1e-6)
        self.hidden_norm = nn.RMSNorm(H, eps=1e-6)
        self.q_proj = nn.Linear(2 * H, nh * hd, bias=False)
        self.k_proj = nn.Linear(2 * H, nkv * hd, bias=False)
        self.v_proj = nn.Linear(2 * H, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, H, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(H, eps=1e-6)
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)


def rope(x, pos, rot_dim=64, theta=1e7):
    # x: (T, heads, hd); partial rotary on the first rot_dim dims (partial_rotary_factor 0.25)
    half = rot_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, device=x.device).float() / rot_dim))
    freqs = pos[:, None].float() * inv[None, :]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rot, xp], dim=-1)


class Eagle3(nn.Module):
    def __init__(self, V, H, I, nh, nkv, hd, embed):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.embed_tokens.weight.data.copy_(embed.float())
        self.fc = nn.Linear(3 * H, H, bias=False)
        self.midlayer = Eagle3Layer(H, I, nh, nkv, hd)
        self.norm = nn.RMSNorm(H, eps=1e-6)
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight.data.copy_(embed.float())
        self.nh, self.nkv, self.hd = nh, nkv, hd

    def forward(self, next_ids, aux, pos):
        m = self.midlayer
        e = self.embed_tokens(next_ids)
        h0 = self.fc(aux)  # residual stream entering the layer
        x = torch.cat([m.input_layernorm(e), m.hidden_norm(h0)], dim=-1)
        T = x.shape[0]
        q = rope(m.q_proj(x).view(T, self.nh, self.hd), pos)
        k = rope(m.k_proj(x).view(T, self.nkv, self.hd), pos)
        v = m.v_proj(x).view(T, self.nkv, self.hd)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        att = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)
        a = m.o_proj(att[0].transpose(0, 1).reshape(T, -1))
        r = a + h0
        mlp = m.down_proj(F.silu(m.gate_proj(m.post_attention_layernorm(r))) * m.up_proj(m.post_attention_layernorm(r)))
        return self.lm_head(self.norm(mlp + r))


def eagle3(args):
    from safetensors.torch import save_file
    blob = torch.load(args.feats, weights_only=False)
    data, embed = blob["data"], blob["embed"]
    V, H = embed.shape
    model = Eagle3(V, H, 6144, 8, 2, 256, embed).cuda()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    # keep the embedding and lm_head trainable but at a small lr so the target init is preserved
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith(("embed_tokens", "lm_head"))], "lr": args.lr},
        {"params": [model.embed_tokens.weight, model.lm_head.weight], "lr": args.lr * 0.1},
    ])
    random.seed(0)
    for ep in range(args.epochs):
        random.shuffle(data)
        tot, n = 0.0, 0
        for d in data:
            ids, plen = d["ids"].cuda(), d["plen"]
            aux = d["aux"].cuda().float()
            T = ids.shape[0]
            t = torch.arange(max(plen - 1, 0), T - 2, device="cuda")
            if t.numel() == 0:
                continue
            pos = torch.arange(T, device="cuda")
            logits_all = model(ids[1:], aux[:-1], pos[:-1])  # position t uses token t+1 and aux_t
            logits = logits_all[t]
            loss = F.cross_entropy(logits.float(), ids[t + 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            n += 1
        print(f"eagle3 epoch {ep} mean loss {tot / max(n, 1):.4f}", flush=True)

    sd = model.state_dict()
    tensors = {
        "embed_tokens.weight": sd["embed_tokens.weight"],
        "fc.weight": sd["fc.weight"],
        "norm.weight": sd["norm.weight"],
        "lm_head.weight": sd["lm_head.weight"],
    }
    for n in ["input_layernorm.weight", "hidden_norm.weight", "post_attention_layernorm.weight",
              "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
              "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
        key = "midlayer." + n
        local = n.replace("self_attn.", "").replace("mlp.", "")
        tensors[key] = sd["midlayer." + local]
    tensors = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in tensors.items()}
    out = os.path.join(args.out, "eagle3")
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    cfg = {
        "architectures": ["Eagle3LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": H, "intermediate_size": 6144, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 1,
        "vocab_size": V, "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
        "rope_theta": 10000000, "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
        "hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "draft_vocab_size": V,
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print("eagle3 saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "medusa", "eagle3"])
    ap.add_argument("--model")
    ap.add_argument("--feats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_pos", type=int, default=0, help="Medusa: max positions per sequence per head (0 = all)")
    ap.add_argument("--train_file", default=TRAIN, help="extract: training prompts (train.jsonl = 256, train1000.jsonl = 1000)")
    args = ap.parse_args()
    if args.stage == "extract":
        extract(args)
    elif args.stage == "medusa":
        medusa(args)
    else:
        eagle3(args)


if __name__ == "__main__":
    main()
