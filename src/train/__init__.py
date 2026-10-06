"""Training (step 10 onwards): one LoRA run per GPU over a frozen step-9 training file. Pure modules (config, data,
grpo_logic) are unit-tested without a GPU; run.py and merge.py import torch, transformers, peft and trl inside main()."""
