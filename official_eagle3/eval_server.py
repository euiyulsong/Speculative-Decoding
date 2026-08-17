#!/usr/bin/env python3
"""Evaluate a running vLLM OpenAI server and read speculative-decoding metrics.

Run a target or EAGLE server, then call this script.  When EAGLE is enabled it
tries both current and legacy Prometheus counter names and reports acceptance.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import re
import time
import urllib.request
from urllib.error import URLError

from openai import OpenAI
from transformers import AutoTokenizer
from common import exact_match, best_f1


def metric(text, base_name):
    # Accept both `name` and Prometheus `_total` variants, with or without labels.
    pat = re.compile(r"^" + re.escape(base_name) + r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.M)
    vals = [float(x) for x in pat.findall(text)]
    return sum(vals) if vals else None


def get_metrics(base_url):
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    try:
        return urllib.request.urlopen(url + "/metrics", timeout=5).read().decode("utf-8", "replace")
    except (URLError, TimeoutError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--target-tokenizer", default="./target_sft")
    ap.add_argument("--eval", default="eval.jsonl")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    client = OpenAI(base_url=args.url, api_key="dummy")
    if args.model is None:
        args.model = client.models.list().data[0].id

    tok = AutoTokenizer.from_pretrained(args.target_tokenizer)
    with open(args.eval, encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()][: args.n_eval]

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        ) for r in rows
    ]

    # Warmup
    client.completions.create(model=args.model, prompt=prompts[0], temperature=0, max_tokens=args.max_new_tokens)
    before = get_metrics(args.url)

    em = f1 = 0.0
    t0 = time.perf_counter()
    for row, prompt in zip(rows, prompts):
        out = client.completions.create(model=args.model, prompt=prompt, temperature=0, max_tokens=args.max_new_tokens)
        pred = out.choices[0].text.strip().split("\n")[0]
        if pred.lower().startswith("answer:"):
            pred = pred[7:].strip()
        em += exact_match(pred, row["answers"])
        f1 += best_f1(pred, row["answers"])
    elapsed = time.perf_counter() - t0
    after = get_metrics(args.url)

    print(f"EM   : {100*em/len(rows):.2f}")
    print(f"F1   : {100*f1/len(rows):.2f}")
    print(f"ms/q : {1000*elapsed/len(rows):.2f}")

    accepted_name = "vllm:spec_decode_num_accepted_tokens"
    drafted_name = "vllm:spec_decode_num_draft_tokens"
    a0, a1 = metric(before, accepted_name), metric(after, accepted_name)
    d0, d1 = metric(before, drafted_name), metric(after, drafted_name)
    if None not in (a0, a1, d0, d1) and d1 > d0:
        accepted = a1 - a0
        drafted = d1 - d0
        print(f"accepted/drafted: {accepted:.0f}/{drafted:.0f}")
        print(f"draft acceptance: {accepted/drafted:.3f}")
    else:
        # Some vLLM versions expose only a gauge.
        rate = metric(after, "vllm:spec_decode_draft_acceptance_rate")
        if rate is not None:
            print(f"draft acceptance gauge: {rate}")
        else:
            print("acceptance: unavailable from this vLLM /metrics build")


if __name__ == "__main__":
    main()
