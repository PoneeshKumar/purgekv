Agent session: 32768 tokens, saved every 1024 tokens (32 saves); 20 sessions/day

| Model | KV/token | KV at 32K | Recompute 32K | Cold resume raw (M3U / M5) | Cold resume 8-bit (M3U / M5) | SSD writes/session: full rewrite | append | append+lossless | append+8-bit | Writes/year (full → append+8-bit) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B (measured on M4 laptop) | 112 KB | 3.8 GB | 99–113 s | 0.8 s / 0.3 s | 0.4 s / 0.2 s | 62.0 GB | 3.8 GB | 3.2 GB | 2.0 GB | 453 TB → 15 TB |
| Llama-3.3-70B dense | 320 KB | 10.7 GB | 131–328 s | 2.1 s / 0.8 s | 1.1 s / 0.4 s | 177.2 GB | 10.7 GB | 8.0 GB | 5.7 GB | 1293 TB → 42 TB |
| Qwen3-235B-A22B MoE | 188 KB | 6.3 GB | 66–131 s | 1.3 s / 0.5 s | 0.7 s / 0.3 s | 104.1 GB | 6.3 GB | 4.7 GB | 3.3 GB | 760 TB → 24 TB |
| DeepSeek-V3/R1 671B (MLA) | 69 KB | 2.3 GB | 156–218 s | 0.5 s / 0.2 s | 0.2 s / 0.1 s | 38.0 GB | 2.3 GB | 1.7 GB | 1.2 GB | 277 TB → 9 TB |
