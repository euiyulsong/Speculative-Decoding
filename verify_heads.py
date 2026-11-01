"""Load a trained head (Medusa or EAGLE3) into vLLM 0.19.1 on a Qwen3.5-2B target and run
greedy generation on the first N SQuAD prompts. Reports acceptance rate and sample outputs.

Usage: python3 verify_heads.py --target <path> --method medusa|eagle3 --head <head_dir> [--n 10]
"""
import argparse
import json

from vllm_squad_bench import clean, spec_counters

MAX_NEW = 32
K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--method", choices=["medusa", "eagle3"], required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(x) for x in open("eval.jsonl", encoding="utf-8")][: args.n]
    llm = LLM(model=args.target, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, language_model_only=True, disable_log_stats=False,
              speculative_config={"method": args.method, "model": args.head,
                                  "num_speculative_tokens": K})
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    llm.generate([rows[0]["prompt"]], sp, use_tqdm=False)  # warm-up
    before = spec_counters(llm)
    outs, golds = [], []
    for r in rows:
        o = llm.generate([r["prompt"]], sp, use_tqdm=False)[0].outputs[0].text
        outs.append(clean(o))
        golds.append(r["answers"][0])
    after = spec_counters(llm)
    drafts = after["vllm:spec_decode_num_drafts"] - before["vllm:spec_decode_num_drafts"]
    dtok = after["vllm:spec_decode_num_draft_tokens"] - before["vllm:spec_decode_num_draft_tokens"]
    acc = after["vllm:spec_decode_num_accepted_tokens"] - before["vllm:spec_decode_num_accepted_tokens"]
    em = sum(o.strip().lower() == g.strip().lower() for o, g in zip(outs, golds)) / len(rows)
    res = {"target": args.target, "method": args.method, "head": args.head, "n": len(rows),
           "EM_approx": round(100 * em, 1),
           "accept_rate": round(acc / dtok, 3) if dtok else None,
           "accepted_per_draft": round(acc / drafts, 3) if drafts else None,
           "sample_outputs": outs[:3], "gold": golds[:3]}
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
