"""SQuAD 1-shot speculative-decoding benchmark with vLLM (native speculative decoding).

Same prompts, metrics and greedy decoding as benchmark_all.py (EM / F1 / ms per query,
MAX_NEW=32, N=200 eval items from eval.jsonl), but generation runs through vLLM so that the
speculative methods are vLLM's own implementations.

Usage (one method per process, so each engine starts clean):
  python3 vllm_squad_bench.py --method base
  python3 vllm_squad_bench.py --method ngram
  python3 vllm_squad_bench.py --method draft --draft Qwen/Qwen3.5-0.8B
  python3 vllm_squad_bench.py --method mtp
Results are appended to results_vllm_squad.jsonl.
"""
import argparse
import json
import statistics
import time

from common import best_f1, exact_match

K = 4
MAX_NEW = 32
N_EVAL = 200


def clean(text):
    text = text.strip().split("\n")[0]
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    return text


def build_speculative_config(args):
    if args.method == "base":
        return None
    if args.method == "ngram":
        return {"method": "ngram", "num_speculative_tokens": K,
                "prompt_lookup_max": 4, "prompt_lookup_min": 1}
    if args.method == "draft":
        # explicit method: vLLM otherwise rewrites qwen3_5 drafts to the MTP loader (hidden-size mismatch)
        return {"method": "draft_model", "model": args.draft, "num_speculative_tokens": K}
    if args.method == "mtp":
        return {"method": "mtp", "num_speculative_tokens": K}
    raise ValueError(args.method)


def spec_counters(llm):
    """Return (drafts, draft_tokens, accepted_tokens) from vLLM's metrics, or None."""
    vals = {"vllm:spec_decode_num_drafts": 0, "vllm:spec_decode_num_draft_tokens": 0,
            "vllm:spec_decode_num_accepted_tokens": 0}
    for m in llm.get_metrics():
        if m.name in vals:
            vals[m.name] = int(getattr(m, "value", 0))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "ngram", "draft", "mtp"], required=True)
    ap.add_argument("--target", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--draft", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data", default="eval.jsonl")
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--out", default="results_vllm_squad.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[: args.n]

    spec = build_speculative_config(args)
    llm_kwargs = dict(model=args.target, dtype="bfloat16", max_model_len=2048,
                      gpu_memory_utilization=0.85, language_model_only=True,
                      disable_log_stats=False)
    if spec is not None:
        llm_kwargs["speculative_config"] = spec
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)

    # warm-up so compilation / CUDA graphs are not counted in latency
    llm.generate([rows[0]["prompt"]], sp, use_tqdm=False)
    before = spec_counters(llm)

    em_scores, f1_scores, latencies, outputs = [], [], [], []
    for r in rows:
        t0 = time.perf_counter()
        out = llm.generate([r["prompt"]], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        pred = clean(out[0].outputs[0].text)
        outputs.append(pred)
        em_scores.append(exact_match(pred, r["answers"]))
        f1_scores.append(best_f1(pred, r["answers"]))

    after = spec_counters(llm)
    drafts = after["vllm:spec_decode_num_drafts"] - before["vllm:spec_decode_num_drafts"]
    draft_tok = after["vllm:spec_decode_num_draft_tokens"] - before["vllm:spec_decode_num_draft_tokens"]
    acc_tok = after["vllm:spec_decode_num_accepted_tokens"] - before["vllm:spec_decode_num_accepted_tokens"]
    result = {
        "task": "squad_1shot", "method": args.method, "target": args.target,
        "draft": args.draft if args.method == "draft" else None,
        "n": len(rows),
        "EM": 100 * statistics.fmean(em_scores),
        "F1": 100 * statistics.fmean(f1_scores),
        "ms_per_q": statistics.fmean(latencies),
        "accept_rate": (acc_tok / draft_tok) if draft_tok else None,
        "mean_accepted_per_draft": (acc_tok / drafts) if drafts else None,
        "sample_outputs": outputs[:3],
    }
    print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
