# Speculative Decoding

## SQuAD — 256 Training Samples, 1-Shot Evaluation


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

SFT improves EM from 66.0 to 88.5, while speculative decoding preserves the target model's accuracy; MTP achieves the best measured speedup (1.74×).
Fine-tuning the draft substantially improves acceptance (0.430 → 0.804), while EAGLE On-Policy improves acceptance over teacher forcing (0.276 → 0.505).

## SQuAD — 1,000 Training Samples, 1-Shot Evaluation


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
| SFT | 93.00 | 94.47 |  187.56  | 1.00× | 0.000 |
| **EAGLE3-HF-OffPolicy** | **93.00** | **94.47** | **482.06** | **0.39×** | 0.625 |
| **EAGLE3-HF-OnPolicy** | **93.00** | **94.47** | **468.75** | **0.40×** | **0.676** |

| Method | EM | F1 | Latency (ms/q) | Speed |
|---|---:|---:|---:|---:|
| Qwen3.5-2B   |61.00 |64.78  | 953.86 |  1.00x|
| Qwen3.5-2B+MTP | 61.00|64.78  |942.63  | 1.01x |

## Qwen3.5-2B N-gram Speculative Decoding

| Method | Latency (ms/query) ↓ | Throughput (tok/s) ↑ | Speedup ↑ | N-gram Match Rate | Acceptance Rate | Avg. Accepted Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B Cached | 951.37 | 31.21 | 1.00× | – | – | – |
| Qwen3.5-2B + N-gram | **467.28** | **41.28** | **2.04×** | 0.828 | 0.121 | 0.29 |

### Summary

- **N-gram requires no additional training or draft model.**
- N-gram speculative decoding achieved **2.04× speedup** over standard cached decoding.
- Latency decreased from **951.37 ms/query → 467.28 ms/query**.
- Throughput increased from **31.21 → 41.28 tokens/s**.
- An N-gram match was found in **82.8%** of decoding steps.
- Only **12.1%** of proposed tokens were accepted by the target model.
- Despite the relatively low acceptance rate, N-gram decoding achieved substantially lower end-to-end latency in this experiment.

## vLLM Reproduction (vLLM 0.18.1, Qwen3.5-2B, batch size 1, greedy)

These runs use vLLM's built-in speculative decoding (`vllm_squad_bench.py`, `gsm8k_vllm_bench.py`) instead of the HF loops above. Latency is per query, one request at a time, after a warm-up request.

### SQuAD 1-shot (200 eval items, `results_vllm_squad.jsonl`)

| Method | EM | F1 | ms/q | Speed vs base | Accept rate | Accepted / draft |
|---|---:|---:|---:|---:|---:|---:|
| Base | 66.00 | 67.42 | 165.5 | 1.00× | – | – |
| N-gram (K=4) | 65.00 | 67.22 | 165.9 | 1.00× | 0.237 | 0.94 |
| MTP (K=4) | 65.00 | 67.22 | 176.6 | 0.94× | 0.467 | 1.87 |

The base EM (66.00) matches the README's 1-shot base row. MTP has the higher acceptance rate but is slightly slower here.

### GSM8K 1-shot (first 100 test items, `results_vllm_gsm8k.jsonl`)

| Method | Accuracy | ms/q | Speed vs base | Accept rate |
|---|---:|---:|---:|---:|
| Base | 53.0 | 504.9 | 1.00× | – |
| N-gram (K=4) | 26.0 | 472.5 | 1.07× | 0.230 |
| MTP (K=4) | 51.0 | 402.5 | 1.25× | 0.664 |

### Open issues

- **Draft model (Qwen3.5-0.8B) is not reproduced in vLLM.** vLLM 0.18.1 rewrites Qwen3.5 draft configs to the MTP loader, and the hidden sizes do not match (2048 vs 1024). Setting `method: draft_model` does not avoid this.
- **N-gram accuracy drop on GSM8K is not explained.** Greedy speculative output should match the base output, but 79 of 100 items diverge (median divergence at character 59). The base model is deterministic across runs (30/30 identical), and K=1 also diverges (7/30 identical to base on 30 items), so the cause is not only multi-token verification. The likely candidates are bf16 numerical differences between the verify forward and the decode forward, or state handling in Qwen3.5's linear-attention layers. An fp32 comparison has not been run yet.
- **SFT rows are not reproduced.** Full fine-tuning of the 2B target does not fit in a 24 GB GPU with AdamW.
- **Math and code:** GSM8K is run on 100 items only. HumanEval is not yet run with vLLM.

### Reproduce

```bash
# vLLM 0.18.1 with torch 2.10 (CUDA 12.8). vLLM 0.22+ wheels need CUDA 13 drivers.
python3 prepare.py                 # writes train.jsonl / eval.jsonl
bash run_squad_vllm.sh             # base, ngram, draft, mtp on SQuAD
bash run_gsm8k_vllm.sh             # base, ngram, mtp on GSM8K
python3 ngram_k1_check.py 1        # losslessness check, 30 items
python3 base_determinism_check.py  # base run-to-run determinism
```
