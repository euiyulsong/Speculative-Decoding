"""Is greedy base decoding itself deterministic run-to-run (same engine, same prompts)?

If two identical base passes disagree on many items, the speculative mismatch is numerical
(bf16 argmax near-ties), not a speculative-decoding bug.
"""
import pandas as pd

from gsm8k_vllm_bench import GSM_DIR, build_prompt, pred_answer, gold_answer

N = 30


def main():
    from vllm import LLM, SamplingParams

    train = pd.read_parquet(GSM_DIR + "train-00000-of-00001.parquet")
    test = pd.read_parquet(GSM_DIR + "test-00000-of-00001.parquet").iloc[:N]
    demo = train.iloc[0].to_dict()
    llm = LLM(model="Qwen/Qwen3.5-2B", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, language_model_only=True)
    sp = SamplingParams(temperature=0.0, max_tokens=256, stop=["\nQuestion:", "Question:"])
    prompts = [build_prompt(demo, r["question"]) for _, r in test.iterrows()]
    golds = [gold_answer(a) for a in test["answer"]]

    runs = []
    for _ in range(2):
        outs = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
        runs.append(outs)
    same = sum(a == b for a, b in zip(runs[0], runs[1]))
    acc = [sum(pred_answer(o) == g for o, g in zip(run, golds)) / N * 100 for run in runs]
    print(f"BASE batched run1 vs run2 identical: {same}/{N}  accuracy: {acc}")


if __name__ == "__main__":
    main()
