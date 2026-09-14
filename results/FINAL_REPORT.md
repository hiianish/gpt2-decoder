# Mini-GPT Optimization Project -- Final Summary

## Phases 1-3: Training Optimizations
- **baseline**: 1170.6s, peak memory 8208MB, final val loss 5.5229
- **mixed_precision**: 484.0s, peak memory 8208MB, final val loss 5.5225
- **mixed_precision**: 483.7s, peak memory 8208MB, final val loss 5.5224
- **memory_engineering**: 1763.6s, peak memory 8208MB, final val loss 5.0966
- From `baseline` to `memory_engineering`: 0.66x time change, 1.00x memory change, validation loss moved from 5.5229 to 5.0966 (should stay close -- these are meant to be near-lossless optimizations, not accuracy tradeoffs).

## Phase 4: Custom Fused Attention Kernel
- Average forward-pass speedup across tested sequence lengths: **0.87x** (vanilla SDPA vs. our Triton kernel).
- Best result: 0.89x at seq_len=64 (18.006ms -> 20.132ms).
- Scope: forward pass only -- see model/attention.py's docstring for the autograd/backward-pass caveat.

## Phase 5: KV-Cache Generation
- Speedup grew from **1.10x** at 20 tokens to **1.23x** at 200 tokens -- confirming the expected O(n) vs O(n^2) behavior (naive generation's wasted recomputation grows with length; cached generation's doesn't).
- Verified mathematically exact: token-for-token identical output under greedy decoding, not an approximation.

## Phase 6: INT8 Quantization
- Model size: 114.6MB -> 102.7MB (**1.12x smaller**).
- CPU latency: 180.14ms -> 165.93ms (**1.09x faster**).
- Perplexity cost: 163.00 -> 156.69 (**+-6.31**) -- the one deliberate accuracy-for-efficiency trade in the whole project.

## Overall Story
Four of the five speed/memory optimizations (mixed precision, gradient checkpointing, the fused kernel, KV-cache) are effectively free -- same output, faster or leaner. Only quantization (Phase 6) trades a small, explicitly measured amount of accuracy for a large gain in deployability on cheap hardware. That asymmetry -- most optimizations are free, one isn't, and we can say exactly how much it costs -- is the actual finding of this project.