# Mini-GPT: Training & Inference Optimization

A small GPT-style decoder-only transformer, trained from scratch and optimized through five performance-focused techniques plus one deliberate quantization tradeoff. Each phase is benchmarked to measure its actual impact on speed, memory, or model quality.


**Compute:** Google Colab · Free T4 GPU  
**Model:** ~30M parameter decoder-only transformer  
**Data:** ~100MB OpenWebText slice · GPT-2 BPE tokenizer

## The Idea

Build the model from scratch, establish a baseline, and then progressively apply optimization techniques to improve training efficiency and inference performance.# Mini-GPT: Training & Inference Optimization

A small GPT-style decoder-only transformer, trained from scratch and optimized through five performance-focused techniques plus one deliberate quantization tradeoff. Each phase is benchmarked to measure its actual impact on speed, memory, or model quality.

**Compute:** Google Colab · Free T4 GPU  
**Model:** ~30M parameter decoder-only transformer  
**Data:** ~100MB OpenWebText slice · GPT-2 BPE tokenizer

## The Idea

Build the model from scratch, establish a baseline, and then progressively apply optimization techniques to improve training efficiency and inference performance.

The project focuses on measuring the effect of each optimization rather than assuming that an optimization is beneficial.

Five optimizations aim to improve efficiency while preserving model behavior, while INT8 quantization introduces the project's only intentional accuracy-versus-efficiency tradeoff.

## Phases

| # | Phase | Notebook / Script | What It Measures |
|---|---|---|---|
| 1–3 | Baseline → Mixed Precision → Memory Engineering | Training notebooks | Measures the effect of AMP, gradient checkpointing, and gradient accumulation on training time and GPU memory while monitoring loss |
| 4 | Custom Fused Attention Kernel | Triton attention implementation | Benchmarks a hand-written Triton attention kernel against PyTorch SDPA for forward-pass latency |
| 5 | KV-Cache Generation | `generate.py` | Verifies cached generation against naive generation and measures the speed advantage as generated sequence length increases |
| 6 | INT8 Quantization | `quantize.py` | Compares FP32 and INT8 inference in terms of model size, CPU inference speed, and perplexity |
| — | Final Benchmark Report | `benchmark.py` | Combines results from all phases into a unified benchmark report and figure |
| — | Hugging Face Upload | `upload_to_hf.py` | Packages and publishes the trained model with `config.json`, `model.safetensors`, and a model card |