import copy
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Config
# ============================================================

MODEL_NAME = "Qwen/Qwen3.5-2B"

DEVICE = "cuda"
DTYPE = torch.bfloat16

N_EVAL = 100
MAX_NEW_TOKENS = 32

K = 4
MAX_NGRAM = 4


# ============================================================
# Dataset
# ============================================================

print("Loading SQuAD...")

dataset = load_dataset(
    "rajpurkar/squad",
    split=f"validation[:{N_EVAL}]",
)

print("Examples:", len(dataset))


# ============================================================
# Model
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
).to(DEVICE)

model.eval()


# ============================================================
# Prompt
# ============================================================

def make_prompt(row):
    return f"""Answer the question using only the context.
Give a short answer.

Context:
{row["context"]}

Question:
{row["question"]}

Answer:"""


prompts = [
    make_prompt(row)
    for row in dataset
]


# ============================================================
# N-gram proposer
# ============================================================

def ngram_propose(prefix_ids, k=K, max_ngram=MAX_NGRAM):
    """
    Find longest suffix appearing previously in the prefix.

    Example:

        ... A B C X Y Z ... A B C

    -> propose X Y Z
    """

    tokens = prefix_ids[0].tolist()
    n_tokens = len(tokens)

    if n_tokens < 2:
        return []

    for n in range(
        min(max_ngram, n_tokens - 1),
        0,
        -1,
    ):
        suffix = tokens[-n:]

        # latest previous occurrence first
        for i in range(n_tokens - n - 1, -1, -1):

            if tokens[i:i + n] == suffix:

                start = i + n

                candidates = tokens[
                    start:start + k
                ]

                if candidates:
                    return candidates

    return []


# ============================================================
# Cached normal generation
# ============================================================

@torch.inference_mode()
def normal_cached_generate(prompt):

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500,
    ).to(DEVICE)

    prefix = enc.input_ids
    original_len = prefix.shape[1]

    # --------------------------------
    # Prefill ONCE
    # --------------------------------

    out = model(
        input_ids=prefix,
        use_cache=True,
    )

    cache = out.past_key_values

    next_token = (
        out.logits[:, -1]
        .argmax(-1)
    )

    generated = []

    while len(generated) < MAX_NEW_TOKENS:

        token = next_token.item()
        generated.append(token)

        if token == tokenizer.eos_token_id:
            break

        # --------------------------------
        # Decode ONE token using cache
        # --------------------------------

        out = model(
            input_ids=next_token[:, None],
            past_key_values=cache,
            use_cache=True,
        )

        cache = out.past_key_values

        next_token = (
            out.logits[:, -1]
            .argmax(-1)
        )

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


# ============================================================
# Cache branch
# ============================================================

def clone_cache(cache):
    """
    Qwen3.5 cache contains both attention KV states
    and linear-attention recurrent states.

    deepcopy gives us an independent speculative branch.
    """
    return copy.deepcopy(cache)


# ============================================================
# N-gram cached generation
# ============================================================

@torch.inference_mode()
def ngram_cached_generate(prompt):

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500,
    ).to(DEVICE)

    prefix_ids = enc.input_ids
    original_len = prefix_ids.shape[1]

    # ========================================================
    # Prefill ONCE
    # ========================================================

    out = model(
        input_ids=prefix_ids,
        use_cache=True,
    )

    cache = out.past_key_values

    next_token = (
        out.logits[:, -1]
        .argmax(-1)
        .item()
    )

    generated = []

    # Stats
    total_steps = 0
    matched_steps = 0

    proposed_total = 0
    accepted_total = 0

    accepted_chunks = []


    while len(generated) < MAX_NEW_TOKENS:

        total_steps += 1

        # ====================================================
        # First target token is already available
        # ====================================================

        current_target_token = next_token

        # ====================================================
        # Try n-gram proposal
        # ====================================================

        proposals = ngram_propose(
            prefix_ids,
            k=K,
            max_ngram=MAX_NGRAM,
        )

        # ====================================================
        # No match -> regular cached decoding
        # ====================================================

        if not proposals:

            generated.append(
                current_target_token
            )

            prefix_ids = torch.cat(
                [
                    prefix_ids,
                    torch.tensor(
                        [[current_target_token]],
                        device=DEVICE,
                    ),
                ],
                dim=1,
            )

            if current_target_token == tokenizer.eos_token_id:
                break

            token_tensor = torch.tensor(
                [[current_target_token]],
                device=DEVICE,
            )

            out = model(
                input_ids=token_tensor,
                past_key_values=cache,
                use_cache=True,
            )

            cache = out.past_key_values

            next_token = (
                out.logits[:, -1]
                .argmax(-1)
                .item()
            )

            continue


        # ====================================================
        # We have an n-gram match
        # ====================================================

        matched_steps += 1

        remaining = (
            MAX_NEW_TOKENS
            - len(generated)
        )

        proposals = proposals[:remaining]

        proposed_total += len(proposals)


        # ====================================================
        # First proposal can be compared with logits
        # already computed from current prefix.
        # ====================================================

        accepted = 0
        output_tokens = []


        # ----------------------------------------------------
        # proposal[0]
        # ----------------------------------------------------

        if proposals[0] != current_target_token:

            # immediate rejection

            output_tokens = [
                current_target_token
            ]

        else:

            accepted = 1
            output_tokens.append(
                proposals[0]
            )

            # =================================================
            # Speculative cache branch
            # =================================================

            branch_cache = clone_cache(cache)

            if len(proposals) > 1:

                # Feed proposal[0 ... K-2].
                #
                # Output logits correspond to targets for
                # proposal[1 ... K-1].
                verify_input = torch.tensor(
                    [proposals[:-1]],
                    dtype=torch.long,
                    device=DEVICE,
                )

                branch_out = model(
                    input_ids=verify_input,
                    past_key_values=branch_cache,
                    use_cache=True,
                )

                verify_logits = (
                    branch_out.logits
                    .argmax(-1)[0]
                    .tolist()
                )

                branch_cache = (
                    branch_out.past_key_values
                )

                rejected = False

                for i in range(
                    1,
                    len(proposals),
                ):

                    target_token = (
                        verify_logits[i - 1]
                    )

                    proposed_token = (
                        proposals[i]
                    )

                    if target_token == proposed_token:

                        accepted += 1
                        output_tokens.append(
                            proposed_token
                        )

                    else:

                        output_tokens.append(
                            target_token
                        )

                        rejected = True
                        break

            else:
                rejected = False


            # =================================================
            # ALL proposals accepted
            # =================================================

            if accepted == len(proposals):

                # Need logits after final accepted proposal
                last_proposal = torch.tensor(
                    [[proposals[-1]]],
                    device=DEVICE,
                )

                bonus_out = model(
                    input_ids=last_proposal,
                    past_key_values=branch_cache,
                    use_cache=True,
                )

                branch_cache = (
                    bonus_out.past_key_values
                )

                bonus_token = (
                    bonus_out.logits[:, -1]
                    .argmax(-1)
                    .item()
                )

                if len(output_tokens) < remaining:
                    output_tokens.append(
                        bonus_token
                    )

                # Branch is now valid.
                cache = branch_cache

                next_token = bonus_token


            # =================================================
            # Rejection happened
            # =================================================

            else:

                # Discard branch_cache.
                #
                # Replay only the accepted/correct tokens
                # onto the original cache.

                replay = torch.tensor(
                    [output_tokens],
                    dtype=torch.long,
                    device=DEVICE,
                )

                replay_out = model(
                    input_ids=replay,
                    past_key_values=cache,
                    use_cache=True,
                )

                cache = (
                    replay_out.past_key_values
                )

                next_token = (
                    replay_out.logits[:, -1]
                    .argmax(-1)
                    .item()
                )


        # ====================================================
        # Update stats
        # ====================================================

        accepted_total += accepted
        accepted_chunks.append(accepted)


        # ====================================================
        # Append generated tokens
        # ====================================================

        output_tokens = output_tokens[
            :MAX_NEW_TOKENS - len(generated)
        ]

        generated.extend(
            output_tokens
        )

        prefix_ids = torch.cat(
            [
                prefix_ids,
                torch.tensor(
                    [output_tokens],
                    dtype=torch.long,
                    device=DEVICE,
                ),
            ],
            dim=1,
        )

        if tokenizer.eos_token_id in output_tokens:
            break


    match_rate = (
        matched_steps / total_steps
        if total_steps
        else 0.0
    )

    acceptance = (
        accepted_total / proposed_total
        if proposed_total
        else 0.0
    )

    avg_accepted = (
        sum(accepted_chunks)
        / len(accepted_chunks)
        if accepted_chunks
        else 0.0
    )


    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    stats = {
        "match_rate": match_rate,
        "acceptance": acceptance,
        "avg_accepted": avg_accepted,
    }

    return text, stats


# ============================================================
# Benchmark
# ============================================================

def benchmark(
    name,
    generation_fn,
):

    print(f"\nRunning: {name}")

    # Warmup
    for i in range(3):
        generation_fn(prompts[i])

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    outputs = []

    total_generated_tokens = 0

    match_sum = 0
    acceptance_sum = 0
    avg_accepted_sum = 0

    has_stats = False


    for i, prompt in enumerate(prompts):

        result = generation_fn(prompt)

        if isinstance(result, tuple):

            text, stats = result

            has_stats = True

            match_sum += stats[
                "match_rate"
            ]

            acceptance_sum += stats[
                "acceptance"
            ]

            avg_accepted_sum += stats[
                "avg_accepted"
            ]

        else:

            text = result


        outputs.append(text)

        tokens = tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

        total_generated_tokens += len(tokens)


        if (i + 1) % 10 == 0:

            print(
                f"  {i + 1}/{len(prompts)}"
            )


    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )


    return {

        "name":
            name,

        "ms_per_query":
            elapsed
            / len(prompts)
            * 1000,

        "tok_s":
            total_generated_tokens
            / elapsed,

        "peak_gb":
            torch.cuda.max_memory_allocated()
            / 1024**3,

        "match_rate":
            match_sum / len(prompts)
            if has_stats
            else 0,

        "acceptance":
            acceptance_sum / len(prompts)
            if has_stats
            else 0,

        "avg_accepted":
            avg_accepted_sum / len(prompts)
            if has_stats
            else 0,

        "outputs":
            outputs,
    }


# ============================================================
# Run
# ============================================================

normal_result = benchmark(
    "Qwen3.5-2B Cached",
    normal_cached_generate,
)


ngram_result = benchmark(
    "Qwen3.5-2B + NGram",
    ngram_cached_generate,
)


# ============================================================
# Results
# ============================================================

baseline_ms = normal_result[
    "ms_per_query"
]


print()

print("=" * 110)

print(
    f"{'Method':<27}"
    f"{'ms/q':>12}"
    f"{'tok/s':>12}"
    f"{'Speed':>10}"
    f"{'Match':>10}"
    f"{'Accept':>10}"
    f"{'AvgAcc':>10}"
)

print("-" * 110)


for r in [
    normal_result,
    ngram_result,
]:

    speed = (
        baseline_ms
        / r["ms_per_query"]
    )

    print(
        f"{r['name']:<27}"
        f"{r['ms_per_query']:>12.2f}"
        f"{r['tok_s']:>12.2f}"
        f"{speed:>9.2f}x"
        f"{r['match_rate']:>10.3f}"
        f"{r['acceptance']:>10.3f}"
        f"{r['avg_accepted']:>10.2f}"
    )


print("=" * 110)


# ============================================================
# Correctness check
# ============================================================

same = sum(
    x == y
    for x, y in zip(
        normal_result["outputs"],
        ngram_result["outputs"],
    )
)


print(
    f"\nSame output: "
    f"{same}/{N_EVAL} "
    f"({same / N_EVAL * 100:.1f}%)"
)
