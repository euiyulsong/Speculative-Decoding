"""GSM8K (math) 1-shot speculative-decoding benchmark with vLLM.

Same protocol as vllm_squad_bench.py: greedy, one request at a time (latency), vLLM native
speculative decoding (ngram / MTP / base). Accuracy = final number after '####' matches gold.

Usage:
  python3 gsm8k_vllm_bench.py --method base
  python3 gsm8k_vllm_bench.py --method ngram
  python3 gsm8k_vllm_bench.py --method mtp
Results are appended to results_vllm_gsm8k.jsonl.
"""
import argparse
import glob
import json
import re
import statistics
import time

import pandas as pd

from vllm_squad_bench import build_speculative_config, spec_counters

MAX_NEW = 256
N_EVAL = 100
GSM_DIR = glob.glob("/home/euiyul/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/")[0]


def gold_answer(answer_text):
    return answer_text.split("####")[-1].strip().replace(",", "")


def pred_answer(text):
    if "####" in text:
        tail = text.split("####")[1]
    else:
        tail = text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    return nums[0].replace(",", "").rstrip(".")


def build_prompt(demo, q):
    return (
        "Solve the math word problem. Show brief reasoning, then give the final answer after '####'.\n\n"
        f"Question: {demo['question']}\nAnswer: {demo['answer']}\n\n"
        f"Question: {q}\nAnswer:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "ngram", "mtp", "draft"], required=True)
    ap.add_argument("--target", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--draft", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--out", default="results_vllm_gsm8k.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    train = pd.read_parquet(GSM_DIR + "train-00000-of-00001.parquet")
    test = pd.read_parquet(GSM_DIR + "test-00000-of-00001.parquet").iloc[: args.n]
    demo = train.iloc[0].to_dict()

    class _A:  # reuse the speculative-config builder from the SQuAD script
        method = args.method
        draft = args.draft

    spec = build_speculative_config(_A)
    kwargs = dict(model=args.target, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.85, language_model_only=True, disable_log_stats=False)
    if spec is not None:
        kwargs["speculative_config"] = spec
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, stop=["\nQuestion:", "Question:"])

    llm.generate([build_prompt(demo, test.iloc[0]["question"])], sp, use_tqdm=False)  # warm-up
    before = spec_counters(llm)

    correct, latencies, outputs = [], [], []
    for _, r in test.iterrows():
        prompt = build_prompt(demo, r["question"])
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        text = out[0].outputs[0].text
        outputs.append(text)
        p = pred_answer(text)
        g = gold_answer(r["answer"])
        correct.append(float(p is not None and p == g))

    after = spec_counters(llm)
    drafts = after["vllm:spec_decode_num_drafts"] - before["vllm:spec_decode_num_drafts"]
    draft_tok = after["vllm:spec_decode_num_draft_tokens"] - before["vllm:spec_decode_num_draft_tokens"]
    acc_tok = after["vllm:spec_decode_num_accepted_tokens"] - before["vllm:spec_decode_num_accepted_tokens"]
    result = {
        "task": "gsm8k_1shot", "method": args.method, "target": args.target,
        "n": len(test),
        "accuracy": 100 * statistics.fmean(correct),
        "ms_per_q": statistics.fmean(latencies),
        "accept_rate": (acc_tok / draft_tok) if draft_tok else None,
        "mean_accepted_per_draft": (acc_tok / drafts) if drafts else None,
        "sample_output": outputs[0][:300],
    }
    print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    # per-item predictions, for checking greedy agreement between methods
    with open(f"preds_gsm8k_{args.method}.jsonl", "w", encoding="utf-8") as f:
        for text, c in zip(outputs, correct):
            f.write(json.dumps({"output": text, "correct": c}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
