"""Exactness check for draft-model speculative decoding on vLLM 0.19.1.

--method base  : target alone, greedy; saves raw outputs
--method draft : target + Qwen3.5-0.8B draft, greedy; saves raw outputs
Then compare the two files: greedy speculative decoding should reproduce the base outputs.

Needs VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1 and VLLM_DRAFT_RELAX_KV_GROUP=1 for the draft run.
"""
import argparse
import json

MAX_NEW = 32
K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "draft"], required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(x) for x in open("eval.jsonl", encoding="utf-8")][: args.n]
    kwargs = dict(model=args.target, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.85, language_model_only=True)
    if args.method == "draft":
        kwargs["speculative_config"] = {"method": "draft_model", "model": args.draft,
                                        "num_speculative_tokens": K}
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    outs = [llm.generate([r["prompt"]], sp, use_tqdm=False)[0].outputs[0].text for r in rows]
    json.dump(outs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print("SAVED", args.out, len(outs), flush=True)


if __name__ == "__main__":
    main()
