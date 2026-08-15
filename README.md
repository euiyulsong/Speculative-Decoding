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
