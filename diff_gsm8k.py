import json
import re

base = [json.loads(l) for l in open("preds_gsm8k_base.jsonl", encoding="utf-8")]
ngram = [json.loads(l) for l in open("preds_gsm8k_ngram.jsonl", encoding="utf-8")]

same = sum(b["output"] == n["output"] for b, n in zip(base, ngram))
print("identical outputs:", same, "/", len(base))

# prefix length until first divergence (characters)
def common_prefix(a, b):
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i

pref = [common_prefix(b["output"], n["output"]) for b, n in zip(base, ngram)]
diverged = [p for b, n, p in zip(base, ngram, pref) if b["output"] != n["output"]]
print("items that diverge:", len(diverged))
if diverged:
    s = sorted(diverged)
    print("divergence char position: min", s[0], "median", s[len(s)//2], "max", s[-1])

print("base correct:", sum(b["correct"] for b in base), " ngram correct:", sum(n["correct"] for n in ngram))
flip_to_wrong = [i for i,(b,n) in enumerate(zip(base,ngram)) if b["correct"] and not n["correct"]]
flip_to_right = [i for i,(b,n) in enumerate(zip(base,ngram)) if not b["correct"] and n["correct"]]
print("base-right->ngram-wrong:", len(flip_to_wrong), " base-wrong->ngram-right:", len(flip_to_right))

# show 3 examples of divergence where base is right and ngram is wrong
for i in flip_to_wrong[:3]:
    p = pref[i]
    print("\n--- item", i, "diverges at char", p)
    print("BASE :", repr(base[i]["output"][max(0,p-60):p+120]))
    print("NGRAM:", repr(ngram[i]["output"][max(0,p-60):p+120]))

# do outputs contain a repeated/garbage pattern in ngram?
rep = sum(1 for n in ngram if n["output"].count("\n") > 15 or len(n["output"]) > 900)
print("\nngram outputs that look long/repetitive:", rep)
lens_b = [len(b["output"]) for b in base]
lens_n = [len(n["output"]) for n in ngram]
print("mean output chars base:", sum(lens_b)/len(lens_b), " ngram:", sum(lens_n)/len(lens_n))
