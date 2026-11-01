"""At the first divergent token between base and n-gram greedy outputs, check the base model's
own distribution at that position.

- gap = logprob(base top-1) - logprob(n-gram token). Small gap (< ~0.05 nats) => near-tie, so
  bf16 numerics can flip the argmax. Large gap => the n-gram token was a clearly worse choice
  for the base model, which points at an implementation problem (e.g. wrong acceptance/state).
- If the n-gram token is not in base's top-10 at all, that is strong evidence of a bug.
"""
import json
import statistics

base = json.load(open("tok_base.json", encoding="utf-8"))
ngram = json.load(open("tok_ngram.json", encoding="utf-8"))

gaps, not_in_top10, n_div, same = [], 0, 0, 0
rows = []
for i, (b, n) in enumerate(zip(base, ngram)):
    bt, nt = b["token_ids"], n["token_ids"]
    if bt == nt:
        same += 1
        continue
    n_div += 1
    p = next((k for k in range(min(len(bt), len(nt))) if bt[k] != nt[k]), min(len(bt), len(nt)))
    if p >= len(b["top"]):
        continue
    top = b["top"][p]  # {token_id_str: logprob}, top-10 of the base model at position p
    top1 = max(top.values())
    tok = str(nt[p])
    if tok in top:
        gap = top1 - top[tok]
        gaps.append(gap)
        rows.append((i, p, gap))
    else:
        not_in_top10 += 1
        rows.append((i, p, None))

print(f"items identical: {same}, diverging: {n_div}")
print(f"n-gram token not in base top-10 at divergence: {not_in_top10}")
if gaps:
    gaps_sorted = sorted(gaps)
    print(f"gap (nats) at divergence, n={len(gaps)}: median {statistics.median(gaps):.3f}, "
          f"min {gaps_sorted[0]:.3f}, max {gaps_sorted[-1]:.3f}")
    print("fraction with gap < 0.05:", round(sum(g < 0.05 for g in gaps) / len(gaps), 3))
    print("fraction with gap < 0.5:", round(sum(g < 0.5 for g in gaps) / len(gaps), 3))
print("sample (item, position, gap):", rows[:10])
