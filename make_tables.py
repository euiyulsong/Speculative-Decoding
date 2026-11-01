"""Build the README result tables (vLLM 0.19.1) from the v3 (256-setting) and v4 (1000-setting) files.

Columns: SQuAD EM, SQuAD ms/q, SQuAD speed vs base, SQuAD accept, GSM8K acc, GSM8K ms/q,
GSM8K speed vs base, GSM8K accept.
"""
import json
import sys

ORDER = ["base", "ngram", "ngram_gpu", "suffix", "mtp", "medusa", "eagle3"]


def load(files, setting):
    rows = {}
    for fn, kind in files:
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            t = r["target"]
            if setting == "256":
                name = "zero-shot" if "Qwen3.5-2B" in t and "lora" not in t and "full" not in t else (
                    "LoRA" if "lora2b" in t else "full SFT")
            else:
                name = "LoRA" if "lora1000" in t else "full SFT"
            rows.setdefault((name, r["method"]), {})[kind] = r
    return rows


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def table(rows, targets):
    out = ["| target | method | SQuAD EM | SQuAD ms/q | speed | SQuAD accept | GSM8K acc | GSM8K ms/q | speed | GSM8K accept |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in targets:
        base_s = rows.get((t, "base"), {}).get("squad", {}).get("ms_per_q")
        base_g = rows.get((t, "base"), {}).get("gsm8k", {}).get("ms_per_q")
        for m in ORDER:
            d = rows.get((t, m))
            if not d:
                continue
            s = d.get("squad", {})
            g = d.get("gsm8k", {})
            sp_s = base_s / s["ms_per_q"] if base_s and s.get("ms_per_q") else None
            sp_g = base_g / g["ms_per_q"] if base_g and g.get("ms_per_q") else None
            out.append(
                f"| {t} | {m} | {fmt(s.get('EM'))} | {fmt(s.get('ms_per_q'))} | {fmt(sp_s, 2)}× | "
                f"{fmt(s.get('accept_rate'), 3)} | {fmt(g.get('accuracy'))} | {fmt(g.get('ms_per_q'))} | "
                f"{fmt(sp_g, 2)}× | {fmt(g.get('accept_rate'), 3)} |")
    return "\n".join(out)


if __name__ == "__main__":
    v3 = load([("results_v3_squad.jsonl", "squad"), ("results_v3_gsm8k.jsonl", "gsm8k")], "256")
    v4 = load([("results_v4_squad.jsonl", "squad"), ("results_v4_gsm8k.jsonl", "gsm8k")], "1000")
    print("## 256-setting\n")
    print(table(v3, ["zero-shot", "LoRA", "full SFT"]))
    print("\n## 1000-setting\n")
    print(table(v4, ["LoRA", "full SFT"]))
    sys.stdout.flush()
