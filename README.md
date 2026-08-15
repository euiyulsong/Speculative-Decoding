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
