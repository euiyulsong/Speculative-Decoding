"""Print the v3 sweep table from results_v3_squad.jsonl and results_v3_gsm8k.jsonl.

Target is identified from the path (zero-shot base / LoRA / full SFT)."""
import json
import os

def target_name(path):
    if "lora2b" in path:
        return "lora"
    if "full2b" in path:
        return "full"
    return "zero"

rows = {}
for fname, kind in [("results_v3_squad.jsonl", "squad"), ("results_v3_gsm8k.jsonl", "gsm8k")]:
    if not os.path.exists(fname):
        continue
    for line in open(fname, encoding="utf-8"):
        r = json.loads(line)
        head = os.path.basename(r.get("head") or "") if r.get("head") else ""
        key = (target_name(r["target"]), r["method"])
        rows.setdefault(key, {})[kind] = r

order_t = ["zero", "lora", "full"]
order_m = ["base", "ngram", "ngram_gpu", "suffix", "mtp", "medusa", "eagle3"]
print("| target | method | SQuAD EM | SQuAD ms/q | GSM8K acc | GSM8K ms/q | accept (gsm8k) |")
print("|---|---|---:|---:|---:|---:|---:|")
for t in order_t:
    for m in order_m:
        if (t, m) not in rows:
            continue
        d = rows[(t, m)]
        s = d.get("squad", {})
        g = d.get("gsm8k", {})
        acc = g.get("accept_rate")
        print(f"| {t} | {m} | {s.get('EM', float('nan')):.1f} | {s.get('ms_per_q', float('nan')):.1f} | "
              f"{g.get('accuracy', float('nan')):.1f} | {g.get('ms_per_q', float('nan')):.1f} | "
              f"{'-' if acc is None else f'{acc:.3f}'} |")
