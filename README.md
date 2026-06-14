# BELLS — Burning Expert Logical Loading System

> A llama.cpp fork implementing intelligent MoE expert memory tiering, enabling trillion-parameter models on consumer hardware.

## The Problem

Modern frontier MoE models like DeepSeek V4-Pro (1.6T params) and MiniMax M3 (427B params) require 8+ H100s to run traditionally. The key insight: only a small fraction of parameters activate per token. BELLS exploits this by keeping only hot experts in VRAM, warm experts in RAM, and cold experts on NVMe.

## How It Works

BELLS implements a 3-tier memory hierarchy for MoE expert weights:

| Tier | Storage | Access Speed | Contents |
|------|---------|--------------|----------|
| L1 | VRAM | ~1TB/s | Burning experts (pinned permanently) |
| L2 | RAM | ~50GB/s | Hot experts (pre-loaded) |
| L3 | NVMe | ~3-10GB/s | Cold experts (mmap on demand) |

A learned router predicts which experts will be needed next and promotes them up the hierarchy before they're required, hiding latency behind compute.

## Roadmap

- [ ] **Stage 1** — Expert activation profiling + static VRAM pinning
- [ ] **Stage 2** — Dynamic RAM/VRAM swapping with LRU eviction  
- [ ] **Stage 3** — Learned prefetch router
- [ ] **Stage 4** — Cross-request expert weight sharing
- [ ] **Stage 5** — Expert-aware request batching and scheduling

## Performance Targets (DeepSeek V4-Flash, RTX 2060)

| Stage | Est. t/s |
|-------|----------|
| Baseline CPU | 0.5 t/s |
| Stage 1 | 1-2 t/s |
| Stage 2 | 2-4 t/s |
| Stage 3 | 5-15 t/s |

## Based On

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [antirez/llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash) for DeepSeek V4 architecture support

## License

MIT
