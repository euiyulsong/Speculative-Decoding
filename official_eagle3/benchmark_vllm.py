#!/usr/bin/env python3
"""Fair EAGLE-3 inference benchmark using vLLM's real speculative decoder.

Compares the same SFT verifier with:
  1) no speculation
  2) official off-policy EAGLE-3 checkpoint
  3) official on-policy EAGLE-3 checkpoint

This does NOT use the hand-written verify() loop in benchmark_all.py; verification
is performed by vLLM's EAGLE-3 implementation.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gc
import json
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import exact_match, best_f1


def clean(text: str) -> str:
    text = text.strip().split("\n")[0]
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    return text


def load_eval(path: str, n: int):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()][:n]


def render_prompts(rows, tokenizer):
    if tokenizer.chat_template is None:
        raise RuntimeError("target tokenizer has no chat_template")
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def evaluate(name, model_path, rows, prompts, sampling, speculator=None, k=4):
    kwargs = dict(model=model_path, trust_remote_code=True)
    if speculator:
        kwargs["speculative_config"] = {
            "model": speculator,
            "method": "eagle3",
            "num_speculative_tokens": k,
        }
    llm = LLM(**kwargs)

    # warmup outside the timed region
    llm.generate([prompts[0]], sampling, use_tqdm=False)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - t0

    em = 0.0
    f1 = 0.0
    generated_tokens = 0
    predictions = []
    for row, out in zip(rows, outputs):
        pred = clean(out.outputs[0].text)
        predictions.append(pred)
        em += exact_match(pred, row["answers"])
        f1 += best_f1(pred, row["answers"])
        generated_tokens += len(out.outputs[0].token_ids)

    result = {
        "method": name,
        "EM": 100.0 * em / len(rows),
        "F1": 100.0 * f1 / len(rows),
        "ms_per_q": 1000.0 * elapsed / len(rows),
        "tok_per_s": generated_tokens / elapsed,
        "generated_tokens": generated_tokens,
        "predictions": predictions,
    }

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="./target_sft")
    ap.add_argument("--off", default="official_eagle3/ckpt_off/checkpoint_best")
    ap.add_argument("--on", default="official_eagle3/ckpt_on/checkpoint_best")
    ap.add_argument("--eval", default="eval.jsonl")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    rows = load_eval(args.eval, args.n_eval)
    tok = AutoTokenizer.from_pretrained(args.target)
    prompts = render_prompts(rows, tok)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=42)

    results = []
    results.append(evaluate("SFT-vLLM", args.target, rows, prompts, sampling))
    results.append(evaluate("EAGLE3-OffPolicy", args.target, rows, prompts, sampling, args.off, args.k))
    results.append(evaluate("EAGLE3-OnPolicy", args.target, rows, prompts, sampling, args.on, args.k))

    baseline = results[0]["ms_per_q"]
    print()
    print(f"{'Method':<24}{'EM':>8}{'F1':>8}{'ms/q':>12}{'Speed':>10}{'tok/s':>12}")
    print("-" * 74)
    for r in results:
        speed = baseline / r["ms_per_q"]
        print(f"{r['method']:<24}{r['EM']:>8.2f}{r['F1']:>8.2f}{r['ms_per_q']:>12.2f}{speed:>9.2f}x{r['tok_per_s']:>12.2f}")

    # Lossless sanity check under greedy decoding: EAGLE output should equal target output.
    base_preds = results[0]["predictions"]
    for r in results[1:]:
        equal = sum(a == b for a, b in zip(base_preds, r["predictions"]))
        r["same_as_target"] = equal / len(base_preds)
        print(f"{r['method']} exact generated-text equality vs target: {equal}/{len(base_preds)}")

    for r in results:
        r.pop("predictions", None)
    with open("official_eagle3/results_vllm.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
