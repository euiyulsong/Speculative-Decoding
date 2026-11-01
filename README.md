# Speculative Decoding on Qwen3.5-2B (vLLM 0.19.1)

Speculative decoding methods compared on SQuAD (1-shot, extractive QA) and GSM8K (math, 1-shot), all measured with vLLM 0.19.1 on one RTX 4090 (24 GB), batch size 1, greedy decoding.

## Summary

- **Speed:** MTP (the model's built-in multi-token head) gives a GSM8K speedup in 4 of 5 target settings (1.05–1.26×; 0.91× for the 1000-prompt full SFT). On SQuAD, suffix decoding is the fastest for the zero-shot target (1.66×), but its gain disappears after fine-tuning (1.05–1.09×).
- **Acceptance:** MTP has the highest acceptance on SQuAD (0.79–0.81 after SFT). Medusa and EAGLE3 heads trained here on 256 or 1000 prompts accept almost nothing (≤ 0.08) and are slower than the base model.
- **Accuracy:** speculative decoding should not change greedy outputs, but n-gram changes GSM8K accuracy (51 → 26) and this is not explained yet (see Open issues).
- **Fine-tuning:** SFT on SQuAD raises SQuAD EM (65 → 90–96) but collapses GSM8K (51 → 3–14), because the training answers are short extractive spans.
- **Draft model (Qwen3.5-0.8B):** not measurable on vLLM 0.19.1 (see Open issues).

## Setup

| item | value |
|---|---|
| target | Qwen/Qwen3.5-2B |
| zero-shot | target without training, 1-shot prompt (one SQuAD example in the prompt) |
| LoRA | LoRA (r=32) SFT on SQuAD prompts (256 or 1000), merged |
| full SFT | all parameters trained (bf16, AdamW), SQuAD prompts (256 or 1000) |
| SQuAD eval | 200 validation items, greedy, max 32 new tokens, EM and F1 |
| GSM8K eval | first 100 test items, greedy, max 256 new tokens, final number after `####` |
| speculative tokens | K = 4 for all methods |
| latency | per query, one request at a time, after a warm-up request; "speed" = base ms / method ms (below 1 = slower) |
| engine | vLLM 0.19.1 with torch 2.10 (CUDA 12.8 wheels, since the driver is CUDA 12.4) |

Methods:
- **base:** no speculation.
- **n-gram / ngram_gpu:** prompt-lookup proposer (CPU and GPU versions).
- **suffix:** suffix-tree proposer over prompt and past outputs (arctic-inference).
- **MTP:** Qwen3.5's own multi-token-prediction head (in the base checkpoint).
- **Medusa:** 4 residual-block heads trained by us on the target's last hidden state.
- **EAGLE3:** a one-layer llama-style head trained by us on auxiliary hidden states (layers 2, 12, 21). Only the Qwen3.5 class was patched in vLLM to expose these (see Reproduce).

## Results A: 256 SQuAD training prompts

| target | method | SQuAD EM | SQuAD ms/q | speed | SQuAD accept | GSM8K acc | GSM8K ms/q | speed | GSM8K accept |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | base | 65.0 | 168.3 | 1.00× | – | 51.0 | 503.7 | 1.00× | – |
| zero-shot | ngram | 65.0 | 169.2 | 1.00× | 0.237 | 26.0 | 483.2 | 1.04× | 0.230 |
| zero-shot | ngram_gpu | 65.0 | 205.7 | 0.82× | 0.156 | 26.0 | 578.2 | 0.87× | 0.199 |
| zero-shot | suffix | 65.0 | 101.1 | 1.66× | 0.789 | 48.0 | 502.2 | 1.00× | 0.339 |
| zero-shot | mtp | 65.0 | 177.5 | 0.95× | 0.465 | 51.0 | 403.2 | 1.25× | 0.664 |
| zero-shot | medusa | 65.0 | 395.1 | 0.43× | 0.017 | 51.0 | 1089.6 | 0.46× | 0.057 |
| zero-shot | eagle3 | 65.0 | 395.5 | 0.43× | 0.018 | 52.0 | 1296.5 | 0.39× | 0.015 |
| LoRA | base | 94.0 | 40.3 | 1.00× | – | 3.0 | 168.7 | 1.00× | – |
| LoRA | ngram | 94.0 | 33.1 | 1.22× | 0.325 | 3.0 | 163.1 | 1.03× | 0.159 |
| LoRA | ngram_gpu | 94.0 | 43.7 | 0.92× | 0.321 | 3.0 | 194.7 | 0.87× | 0.139 |
| LoRA | suffix | 94.0 | 37.8 | 1.07× | 0.468 | 3.0 | 187.1 | 0.90× | 0.235 |
| LoRA | mtp | 94.0 | 52.7 | 0.76× | 0.797 | 3.0 | 155.3 | 1.09× | 0.557 |
| LoRA | medusa | 94.0 | 62.2 | 0.65× | 0.050 | 3.0 | 321.7 | 0.52× | 0.068 |
| LoRA | eagle3 | 94.0 | 81.8 | 0.49× | 0.000 | 4.0 | 416.1 | 0.41× | 0.002 |
| full SFT | base | 90.5 | 41.1 | 1.00× | – | 13.0 | 336.8 | 1.00× | – |
| full SFT | ngram | 90.5 | 33.2 | 1.24× | 0.354 | 5.0 | 325.4 | 1.04× | 0.187 |
| full SFT | ngram_gpu | 90.5 | 43.6 | 0.94× | 0.349 | 5.0 | 388.1 | 0.87× | 0.163 |
| full SFT | suffix | 90.5 | 37.5 | 1.09× | 0.465 | 7.0 | 358.9 | 0.94× | 0.267 |
| full SFT | mtp | 90.5 | 53.2 | 0.77× | 0.806 | 14.0 | 267.2 | 1.26× | 0.637 |
| full SFT | medusa | 90.5 | 65.2 | 0.63× | 0.060 | 14.0 | 688.1 | 0.49× | 0.061 |
| full SFT | eagle3 | 90.5 | 84.2 | 0.49× | 0.000 | 13.0 | 807.5 | 0.42× | 0.004 |

## Results B: 1000 SQuAD training prompts

| target | method | SQuAD EM | SQuAD ms/q | speed | SQuAD accept | GSM8K acc | GSM8K ms/q | speed | GSM8K accept |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | base | 96.0 | 40.9 | 1.00× | – | 3.0 | 122.2 | 1.00× | – |
| LoRA | ngram | 96.0 | 33.8 | 1.21× | 0.309 | 3.0 | 115.1 | 1.06× | 0.273 |
| LoRA | ngram_gpu | 96.0 | 44.9 | 0.91× | 0.306 | 3.0 | 141.6 | 0.86× | 0.246 |
| LoRA | suffix | 96.0 | 37.7 | 1.09× | 0.452 | 3.0 | 114.7 | 1.07× | 0.402 |
| LoRA | mtp | 96.0 | 53.3 | 0.77× | 0.793 | 3.0 | 116.6 | 1.05× | 0.596 |
| LoRA | medusa | 96.0 | 58.3 | 0.70× | 0.077 | 3.0 | 221.2 | 0.55× | 0.081 |
| LoRA | eagle3 | 96.0 | 79.0 | 0.52× | 0.008 | 3.0 | 276.3 | 0.44× | 0.022 |
| full SFT | base | 91.5 | 40.0 | 1.00× | – | 4.0 | 72.4 | 1.00× | – |
| full SFT | ngram | 91.5 | 33.4 | 1.20× | 0.325 | 4.0 | 65.9 | 1.10× | 0.193 |
| full SFT | ngram_gpu | 91.5 | 45.2 | 0.89× | 0.321 | 4.0 | 83.2 | 0.87× | 0.175 |
| full SFT | suffix | 91.5 | 38.1 | 1.05× | 0.439 | 4.0 | 70.7 | 1.02× | 0.368 |
| full SFT | mtp | 91.5 | 52.2 | 0.77× | 0.802 | 4.0 | 79.2 | 0.91× | 0.543 |
| full SFT | medusa | 91.5 | 57.9 | 0.69× | 0.081 | 4.0 | 123.9 | 0.58× | 0.067 |
| full SFT | eagle3 | 91.5 | 79.2 | 0.51× | 0.015 | 4.0 | 153.9 | 0.47× | 0.034 |

Notes on the two tables:
- Every row uses the same target checkpoint for both tasks. Zero-shot has no training, so it is the same model in both settings.
- Acceptance is the fraction of proposed tokens accepted by the target (vLLM's counters).
- Medusa and EAGLE3 numbers reflect our small head training, not the released heads from the literature.

## Open issues

- **Draft model (Qwen3.5-0.8B) is not measurable on vLLM 0.19.1.** Three blockers were found and handled in order: (1) vLLM rewrites Qwen3.5 draft configs to the MTP loader; (2) the draft uses M-RoPE, which vLLM does not support for drafts (worked around by removing the M-RoPE keys, which is equivalent for text-only inputs); (3) the draft mixes linear-attention and full-attention layers, and vLLM's draft path assumes one KV-cache spec, so the linear-attention layers assert `MambaSpec`. Fixing (3) needs per-group draft metadata inside vLLM. Draft rows are therefore omitted.
- **N-gram changes GSM8K outputs.** Greedy speculative decoding should match the base output. On 30 items the base model is deterministic (30/30 identical across runs), but n-gram diverges from base on most items, typically early in the answer. At the first divergence the n-gram token is far less likely under the base model (median gap 2.06 nats; 5 cases outside the base top-10), so it is not a near-tie. This points to an implementation problem in vLLM's n-gram path on this hybrid model (for example state handling in the linear-attention layers). Not yet confirmed.
- **SFT on SQuAD breaks GSM8K.** The training answers are short extractive spans, so the tuned models stop writing reasoning. GSM8K numbers for LoRA and full SFT are therefore not meaningful as math ability.
- **Heads are small and under-trained.** Medusa and EAGLE3 heads are trained on 256 or 1000 prompts for 2–3 epochs. The EAGLE3 LoRA head accepts almost nothing (0.000–0.008), so it may not be learning the target's features correctly. This needs checking before reading the EAGLE3 rows as a method comparison.
- **Small evaluation sets.** 200 SQuAD and 100 GSM8K items, one seed per configuration. Differences of a few points are within noise.
- **Patched vLLM.** Two opt-in patches are applied to the installed vLLM 0.19.1: Qwen3.5 aux-hidden-state exposure (needed for EAGLE3), and the draft-path opt-ins. Both are off by default except where noted in Reproduce.

## Reproduce

```bash
# environment: vLLM 0.19.1 (CUDA 12.8 wheels) + torch 2.10 cu128; training env with transformers >= 5.6 and peft
python3 prepare.py                # 256 SQuAD prompts + 200 eval prompts (train.jsonl, eval.jsonl)
python3 prepare_1000.py           # 1000 SQuAD prompts (train1000.jsonl)
python3 train_lora.py --model Qwen/Qwen3.5-2B --out <lora_dir>      # LoRA SFT (256 by default)
python3 train_full.py --model Qwen/Qwen3.5-2B --out <full_dir>      # full SFT
python3 overlay_ckpt.py <trained_dir> <out_dir>                      # write a complete checkpoint (base config and vision tower)
python3 build_heads.py extract --model <target> --out <dir> [--train_file train1000.jsonl]
python3 build_heads.py medusa  --feats <dir>/feats.pt --out <dir> [--max_pos 64] --epochs 2
python3 build_heads.py eagle3  --feats <dir>/feats.pt --out <dir> --epochs 2
python3 vllm_squad_bench.py  --method base|ngram|ngram_gpu|suffix|mtp|medusa|eagle3 --target <dir> [--head <head_dir>] --n 200
python3 gsm8k_vllm_bench.py  --method <same> --target <dir> [--head <head_dir>]
python3 make_tables.py            # regenerates the tables above from results_v3_* and results_v4_*
```

vLLM patches (under `patches/`) are applied to the installed package:
- `patch_qwen35_eagle.py`: exposes EAGLE3 auxiliary hidden states for Qwen3.5 (needed for EAGLE3).
- `patch_qwen35_draft.py` and `fix_draft_patch_flag.py`: opt-in bypass of the Qwen3.5 MTP rewrite, enabled only with `VLLM_QWEN35_DRAFT_NO_MTP_REWRITE=1`. Do not set it for MTP runs.
- `patch_draft_kvgroup.py` and `patch_draft_kvgroup2.py`: opt-in relaxation of the draft KV-group check (`VLLM_DRAFT_RELAX_KV_GROUP=1`). Not sufficient on its own (see Open issues).

## Appendix A: HF-loop results (before the vLLM reproduction)

These were measured with the repo's own HF `generate` loops (`benchmark_all.py`, `ngram.py`), not vLLM. They are kept for reference, including the draft-model rows that vLLM 0.19.1 cannot run.

### SQuAD — 256 Training Samples, 1-Shot Evaluation

| Method | EM | F1 | ms/q | Speed | Accept |
|---|---:|---:|---:|---:|---:|
| Base+1shot | 66.00 | 68.22 | 1034.36 | 1.49x | 0.000 |
| SFT | 88.50 | 93.72 | 1539.09 | 1.00x | 0.000 |
| Tbase+Dbase | 66.00 | 68.06 | 6567.88 | 0.23x | 0.631 |
| Tsft+Dbase | 88.50 | 93.72 | 1145.44 | 1.34x | 0.430 |
| Tbase+Dsft | 66.00 | 68.06 | 8215.15 | 0.19x | 0.404 |
| Tsft+Dsft | 88.50 | 93.72 | 919.23 | 1.67x | 0.804 |
| MTP | 88.50 | 93.72 | 884.28 | 1.74x | 0.193 |
| EAGLE-TF | 88.50 | 93.72 | 1985.53 | 0.78x | 0.276 |
| EAGLE-OnPolicy | 88.50 | 93.72 | 1958.49 | 0.79x | 0.505 |

SFT improves EM from 66.0 to 88.5, while speculative decoding preserves the target model's accuracy; MTP achieves the best measured speedup (1.74×). Fine-tuning the draft substantially improves acceptance (0.430 → 0.804), while EAGLE On-Policy improves acceptance over teacher forcing (0.276 → 0.505).

### SQuAD — 1,000 Training Samples, 1-Shot Evaluation

| Method | EM | F1 | Latency (ms/q) | Speed | Accept Rate |
|---|---:|---:|---:|---:|---:|
| Base + 1-shot | 61.00 | 64.78 | 938.29 | 0.99× | 0.000 |
| SFT | 93.00 | 94.47 | 931.49 | 1.00× | 0.000 |
| Tbase + Dbase | 61.00 | 64.78 | 6277.51 | 0.15× | 0.619 |
| Tsft + Dbase | 93.00 | 94.47 | 1060.79 | 0.88× | 0.329 |
| Tbase + Dsft | 61.00 | 64.78 | 8126.37 | 0.11× | 0.402 |
| Tsft + Dsft | 93.00 | 94.47 | 831.38 | 1.12× | 0.646 |
| **MTP** | **93.00** | **94.47** | **723.92** | **1.29×** | 0.443 |

| Method | EM | F1 | Latency (ms/q) | Speed | Accept Rate |
|---|---:|---:|---:|---:|---:|
| SFT | 93.00 | 94.47 | 187.56 | 1.00× | 0.000 |
| **EAGLE3-HF-OffPolicy** | **93.00** | **94.47** | **482.06** | **0.39×** | 0.625 |
| **EAGLE3-HF-OnPolicy** | **93.00** | **94.47** | **468.75** | **0.40×** | **0.676** |

| Method | EM | F1 | Latency (ms/q) | Speed |
|---|---:|---:|---:|---:|
| Qwen3.5-2B | 61.00 | 64.78 | 953.86 | 1.00x |
| Qwen3.5-2B+MTP | 61.00 | 64.78 | 942.63 | 1.01x |

### Qwen3.5-2B N-gram Speculative Decoding (HF loop)

| Method | Latency (ms/query) ↓ | Throughput (tok/s) ↑ | Speedup ↑ | N-gram Match Rate | Acceptance Rate | Avg. Accepted Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B Cached | 951.37 | 31.21 | 1.00× | – | – | – |
| Qwen3.5-2B + N-gram | **467.28** | **41.28** | **2.04×** | 0.828 | 0.121 | 0.29 |

This 2.04× n-gram speedup was measured with the HF loop. Under vLLM 0.19.1 the n-gram path gives about 1.0× and changes outputs (see Open issues), so the two results should not be compared directly.

## Appendix B: vLLM 0.18.1 reproduction (earlier, 1-shot, 100–200 items)

Earlier run on vLLM 0.18.1 (CUDA 12.8 build, before the upgrade to 0.19.1). Kept for reference. The numbers differ from the main tables because the engine version, model checkpoint, and sample sizes differ.

### SQuAD 1-shot (200 eval items)

| Method | EM | F1 | ms/q | Speed vs base | Accept rate | Accepted / draft |
|---|---:|---:|---:|---:|---:|---:|
| Base | 66.00 | 67.42 | 165.5 | 1.00× | – | – |
| N-gram (K=4) | 65.00 | 67.22 | 165.9 | 1.00× | 0.237 | 0.94 |
| MTP (K=4) | 65.00 | 67.22 | 176.6 | 0.94× | 0.467 | 1.87 |

### GSM8K 1-shot (first 100 test items)

| Method | Accuracy | ms/q | Speed vs base | Accept rate |
|---|---:|---:|---:|---:|
| Base | 53.0 | 504.9 | 1.00× | – |
| N-gram (K=4) | 26.0 | 472.5 | 1.07× | 0.230 |
| MTP (K=4) | 51.0 | 402.5 | 1.25× | 0.664 |

Earlier open issues from this run are superseded by the Open issues section above. The draft model was not reproduced in this run either.
