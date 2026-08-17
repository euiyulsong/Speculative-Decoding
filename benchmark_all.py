import gc, json, time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import exact_match, best_f1

DEVICE = "cuda"
TARGET_BASE = "Qwen/Qwen3.5-2B"
TARGET_SFT = "./target_sft"
DRAFT_BASE = "Qwen/Qwen3.5-0.8B"
DRAFT_SFT = "./draft_sft"
MTP_PATH = "mtp_squad.pt"
EAGLE_TF_PATH = "eagle_teacher.pt"
EAGLE_ON_PATH = "eagle_onpolicy.pt"
K = 4
MAX_NEW = 32
N_EVAL = 200

with open("eval.jsonl", encoding="utf-8") as f:
    eval_rows = [json.loads(x) for x in f][:N_EVAL]

tokenizer = AutoTokenizer.from_pretrained(TARGET_SFT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class MTP(nn.Module):
    def __init__(self, hidden_size, vocab_size, k):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(k)
        ])
    def forward(self, h):
        h = self.norm(h)
        return [head(h) for head in self.heads]

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
        h = self.feature_proj(torch.cat(hidden_states, dim=-1))
        tok = self.token_embed(input_ids)
        x = self.input_proj(torch.cat([tok, h], dim=-1))
        T = x.shape[1]
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        x = self.decoder(x, mask=mask)
        return self.lm_head(self.norm(x))

def load_lm(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEVICE)
    m.eval()
    return m

@torch.inference_mode()
def normal_generate(model, prompt):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    ).to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

@torch.inference_mode()
def verify(target, prefix, proposals):
    proposed = torch.tensor([proposals], device=DEVICE)
    full = torch.cat([prefix, proposed], dim=1)
    out = target(input_ids=full, use_cache=False)
    L = prefix.shape[1]
    append, accepted = [], 0

    for i, token in enumerate(proposals):
        target_token = out.logits[0, L - 1 + i].argmax().item()
        if target_token == token:
            append.append(token)
            accepted += 1
        else:
            append.append(target_token)
            break

    if accepted == len(proposals):
        bonus = out.logits[0, L + len(proposals) - 1].argmax().item()
        append.append(bonus)

    return append, accepted

@torch.inference_mode()
def draft_propose(draft, prefix):
    work = prefix.clone()
    result = []
    for _ in range(K):
        out = draft(input_ids=work, use_cache=False)
        token = out.logits[:, -1].argmax(-1).item()
        result.append(token)
        work = torch.cat(
            [work, torch.tensor([[token]], device=DEVICE)],
            dim=1,
        )
    return result

@torch.inference_mode()
def mtp_propose(target, mtp, prefix):
    out = target(
        input_ids=prefix,
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.hidden_states[-1][:, -1]
    return [x.argmax(-1).item() for x in mtp(h)[:K]]

@torch.inference_mode()
def eagle_propose(target, eagle, layer_ids, prefix):
    work = prefix.clone()
    result = []
    for _ in range(K):
        out = target(
            input_ids=work,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = [out.hidden_states[i] for i in layer_ids]
        logits = eagle(work, hs)
        token = logits[:, -1].argmax(-1).item()
        result.append(token)
        work = torch.cat(
            [work, torch.tensor([[token]], device=DEVICE)],
            dim=1,
        )
    return result

@torch.inference_mode()
def speculative_generate(target, prompt, proposer):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    )
    prefix = enc.input_ids.to(DEVICE)
    original_len = prefix.shape[1]
    accepted_total = 0
    proposed_total = 0

    while prefix.shape[1] - original_len < MAX_NEW:
        proposals = proposer(prefix)
        proposed_total += len(proposals)

        append, accepted = verify(target, prefix, proposals)
        accepted_total += accepted

        remaining = MAX_NEW - (prefix.shape[1] - original_len)
        append = append[:remaining]

        prefix = torch.cat(
            [prefix, torch.tensor([append], device=DEVICE)],
            dim=1,
        )

        if tokenizer.eos_token_id in append:
            break

    generated = prefix[0, original_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    acceptance = accepted_total / proposed_total if proposed_total else 0.0
    return text, acceptance

def clean(text):
    text = text.strip().split("\n")[0]
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    return text

def evaluate(name, generation_fn):
    generation_fn(eval_rows[0]["prompt"])
    torch.cuda.synchronize()
    start = time.perf_counter()

    em_total = 0.0
    f1_total = 0.0
    acceptance_total = 0.0

    for row in eval_rows:
        result = generation_fn(row["prompt"])
        if isinstance(result, tuple):
            pred, acceptance = result
        else:
            pred, acceptance = result, 0.0

        pred = clean(pred)
        em_total += exact_match(pred, row["answers"])
        f1_total += best_f1(pred, row["answers"])
        acceptance_total += acceptance

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "method": name,
        "EM": 100 * em_total / N_EVAL,
        "F1": 100 * f1_total / N_EVAL,
        "ms": elapsed / N_EVAL * 1000,
        "acceptance": acceptance_total / N_EVAL,
    }

def load_eagle(path):
    c = torch.load(path, map_location="cpu")
    m = Eagle3(c["hidden_size"], c["vocab_size"])
    m.load_state_dict(c["model"])
    m = m.to(DEVICE, dtype=torch.bfloat16)
    m.eval()
    return m, c["layer_ids"]

results = []

base_target = load_lm(TARGET_BASE)
results.append(evaluate(
    "Base+1shot",
    lambda p: normal_generate(base_target, p),
))

sft_target = load_lm(TARGET_SFT)
results.append(evaluate(
    "SFT",
    lambda p: normal_generate(sft_target, p),
))

draft_base = load_lm(DRAFT_BASE)
results.append(evaluate(
    "Tbase+Dbase",
    lambda p: speculative_generate(
        base_target, p,
        lambda prefix: draft_propose(draft_base, prefix)
    ),
))
results.append(evaluate(
    "Tsft+Dbase",
    lambda p: speculative_generate(
        sft_target, p,
        lambda prefix: draft_propose(draft_base, prefix)
    ),
))
del draft_base
gc.collect()
torch.cuda.empty_cache()

draft_sft = load_lm(DRAFT_SFT)
results.append(evaluate(
    "Tbase+Dsft",
    lambda p: speculative_generate(
        base_target, p,
        lambda prefix: draft_propose(draft_sft, prefix)
    ),
))
results.append(evaluate(
    "Tsft+Dsft",
    lambda p: speculative_generate(
        sft_target, p,
        lambda prefix: draft_propose(draft_sft, prefix)
    ),
))
del draft_sft
gc.collect()
torch.cuda.empty_cache()

mtp_ckpt = torch.load(MTP_PATH, map_location="cpu")
mtp = MTP(
    mtp_ckpt["hidden_size"],
    mtp_ckpt["vocab_size"],
    mtp_ckpt["k"],
)
mtp.load_state_dict(mtp_ckpt["model"])
mtp = mtp.to(DEVICE, dtype=torch.bfloat16)
mtp.eval()

results.append(evaluate(
    "MTP",
    lambda p: speculative_generate(
        sft_target, p,
        lambda prefix: mtp_propose(sft_target, mtp, prefix)
    ),
))
del mtp
gc.collect()
torch.cuda.empty_cache()

eagle, layers = load_eagle(EAGLE_TF_PATH)
results.append(evaluate(
    "EAGLE-TF",
    lambda p: speculative_generate(
        sft_target, p,
        lambda prefix: eagle_propose(sft_target, eagle, layers, prefix)
    ),
))
del eagle
gc.collect()
torch.cuda.empty_cache()

eagle, layers = load_eagle(EAGLE_ON_PATH)
results.append(evaluate(
    "EAGLE-OnPolicy",
    lambda p: speculative_generate(
        sft_target, p,
        lambda prefix: eagle_propose(sft_target, eagle, layers, prefix)
    ),
))

baseline_ms = next(x["ms"] for x in results if x["method"] == "SFT")

print()
print(f"{'Method':<20}{'EM':>8}{'F1':>8}{'ms/q':>12}{'Speed':>10}{'Accept':>10}")
print("-" * 70)
for x in results:
    speed = baseline_ms / x["ms"]
    print(
        f"{x['method']:<20}"
        f"{x['EM']:>8.2f}"
        f"{x['F1']:>8.2f}"
        f"{x['ms']:>12.2f}"
        f"{speed:>9.2f}x"
        f"{x['acceptance']:>10.3f}"
    )

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
