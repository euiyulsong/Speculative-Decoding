# Corrected speculative-decoding patch

Reuses: train.jsonl, eval.jsonl, target_sft/, mtp_squad.pt.
Adds: sequential MTP, old parallel-MTP baseline, corrected fast EAGLE-TF, corrected fast EAGLE-OnPolicy.

Run:
```bash
chmod +x run_new.sh
./run_new.sh
```

Important: the sequential MTP reproduces the algorithmic structure documented by vLLM Speculators, but it is trained from scratch rather than initialized from Qwen3.5 native MTP checkpoint tensors. Exact native-weight extraction/stitching is model-specific and normally done by a converter. The corrected EAGLE must be retrained because the old checkpoint required target hidden states at each future draft position.
