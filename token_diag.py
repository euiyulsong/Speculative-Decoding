"""Token-level diagnostic for the GSM8K n-gram mismatch.

Phase 'base': greedy base decoding, saves generated token ids and the top-10 logprobs at every
generated position (the base model's own distribution along its own trajectory).
Phase 'ngram': greedy n-gram speculative decoding, saves generated token ids only.

analyze_token_diag.py then finds the first position where the two token sequences differ and
checks, using the base top-10 at that position, whether the n-gram token was a near-tie
(numerical) or a clear base-model error (suspect implementation).
"""
import argparse
import json

import pandas as pd

from gsm8k_vllm_bench import GSM_DIR, build_prompt

N = 30
MAX_NEW = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "ngram"], required=True)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    train = pd.read_parquet(GSM_DIR + "train-00000-of-00001.parquet")
    test = pd.read_parquet(GSM_DIR + "test-00000-of-00001.parquet").iloc[: args.n]
    demo = train.iloc[0].to_dict()
    kwargs = dict(model="Qwen/Qwen3.5-2B", dtype=args.dtype, max_model_len=2048,
                  gpu_memory_utilization=0.85, language_model_only=True)
    if args.method == "ngram":
        kwargs["speculative_config"] = {"method": "ngram", "num_speculative_tokens": 4,
                                        "prompt_lookup_max": 4, "prompt_lookup_min": 1}
    llm = LLM(**kwargs)
    # no stop strings here: we compare raw token sequences
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW,
                        logprobs=10 if args.method == "base" else None)
    records = []
    for _, r in test.iterrows():
        o = llm.generate([build_prompt(demo, r["question"])], sp, use_tqdm=False)[0].outputs[0]
        rec = {"token_ids": list(o.token_ids)}
        if args.method == "base":
            rec["top"] = [
                {str(tid): (lp.logprob) for tid, lp in pos.items()} for pos in o.logprobs
            ]
        records.append(rec)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print("SAVED", args.out, len(records), flush=True)


if __name__ == "__main__":
    main()
