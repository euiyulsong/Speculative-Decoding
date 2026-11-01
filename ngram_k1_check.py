"""Check whether n-gram speculative decoding is lossless (greedy) for K=1 vs K=4 on GSM8K.

If K=1 reproduces base outputs exactly while K=4 diverges, the multi-token verification
(or state rollback for Qwen3.5's linear-attention layers) is the source of the divergence.
"""
import json
import sys

import pandas as pd

from gsm8k_vllm_bench import GSM_DIR, build_prompt, pred_answer, gold_answer

N = 30


def main():
    from vllm import LLM, SamplingParams

    k = int(sys.argv[1])
    train = pd.read_parquet(GSM_DIR + "train-00000-of-00001.parquet")
    test = pd.read_parquet(GSM_DIR + "test-00000-of-00001.parquet").iloc[:N]
    demo = train.iloc[0].to_dict()
    llm = LLM(model="Qwen/Qwen3.5-2B", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, language_model_only=True,
              speculative_config={"method": "ngram", "num_speculative_tokens": k,
                                  "prompt_lookup_max": 4, "prompt_lookup_min": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=256, stop=["\nQuestion:", "Question:"])
    outs, correct = [], []
    for _, r in test.iterrows():
        text = llm.generate([build_prompt(demo, r["question"])], sp, use_tqdm=False)[0].outputs[0].text
        outs.append(text)
        correct.append(float(pred_answer(text) == gold_answer(r["answer"])))
    base = [json.loads(l)["output"] for l in open("preds_gsm8k_base.jsonl", encoding="utf-8")][:N]
    same = sum(a == b for a, b in zip(outs, base))
    print(f"K={k} identical to base: {same}/{N}  accuracy: {100*sum(correct)/N:.1f}")


if __name__ == "__main__":
    main()
