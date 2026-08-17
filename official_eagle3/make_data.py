#!/usr/bin/env python3
"""Build fair off-policy and on-policy EAGLE-3 datasets from ../train.jsonl.

Both datasets use the exact same user prompt.  The only difference is the
assistant response:
  * off-policy: SQuAD gold answer
  * on-policy : verifier/target-generated answer

On-policy generation is performed on the *chat-templated* prompt so that the
conditioning seen during regeneration matches what Speculators prepare_data.py
will tokenize for training.
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def read_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_conversation(prompt: str, answer: str, idx: int):
    return {
        "id": f"squad-{idx}",
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train.jsonl")
    ap.add_argument("--target", default="./target_sft")
    ap.add_argument("--output-dir", default="official_eagle3/data_raw")
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_rows(args.input)[: args.max_samples]
    out_dir = Path(args.output_dir)

    # Off-policy = original SQuAD gold answer.
    off_rows = [as_conversation(r["prompt"], r["answer"], i) for i, r in enumerate(rows)]
    write_jsonl(out_dir / "offpolicy.jsonl", off_rows)
    print(f"wrote {len(off_rows)} off-policy rows")

    # Import vLLM lazily so users can still build the off-policy file without it.
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.target)
    if tok.chat_template is None:
        raise RuntimeError(
            "Target tokenizer has no chat_template. Official Speculators prepare_data.py "
            "expects conversation data and applies the model chat template."
        )

    rendered_prompts = []
    for r in rows:
        rendered_prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    llm = LLM(model=args.target, trust_remote_code=True)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(rendered_prompts, sampling, use_tqdm=True)

    on_rows = []
    for i, (r, out) in enumerate(zip(rows, outputs)):
        answer = out.outputs[0].text.strip()
        on_rows.append(as_conversation(r["prompt"], answer, i))

    write_jsonl(out_dir / "onpolicy.jsonl", on_rows)
    print(f"wrote {len(on_rows)} on-policy rows")


if __name__ == "__main__":
    main()
