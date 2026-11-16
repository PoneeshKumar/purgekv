# Research log

How purgekv got here: the full investigation, including the ideas that did not work.
Raw JSON for every run is in [`results/`](../results). Reproduce it all with `./research/run_all.sh`.

## Can the macOS kernel reclaim idle LLM KV caches for free?

**Verdict (2026-11-15, after full-power reruns):** the problem is real, and the
mechanism works and is new. But it **does not beat the simple fix**. Writing idle KV
caches to an SSD file and dropping them from RAM was as good or better on every
metric except SSD writes. Every attempt to use purgeable memory to save those writes
lost chats to the kernel.

## The problem (confirmed)
Local LLM servers on a Mac keep idle chats' KV caches in RAM (mlx-lm's
`LRUPromptCache`, and the hot tiers of the servers built on it). With 12 idle 8K-token
chats (3 GB of KV) on a 16 GB MacBook, opening an 8 GB app pushed **3.2–5.3 GB to
swap in all 3 reps**. In one rep the app also became ~10× slower to reach its own
memory. The LLM process's footprint (what jetsam ranks by) was 3.9–4.5 GB.

## The idea
XNU *purgeable memory*: mark a region VOLATILE and the kernel may discard it under
pressure with no swap. Marking it NONVOLATILE again reports whether it survived.
purgekv keeps each idle chat's KV in one purgeable object, with a fallback to SSD or
re-prefill when it's gone.

**Novelty:** the current source of mlx, mlx-lm, llama.cpp, ollama, oMLX, vllm-mlx and
mlx-flash only *reads* purgeable counters. No paper or project I found allocates KV in
purgeable memory. The closest is mzCache (arXiv 2609.01338), which does user-space
eviction on Android. A search can't prove a negative.

## Results — M4 MacBook, 16 GB, macOS 26.6, battery, Low Power Mode **off**
MLX 0.32.4 / mlx-lm 0.32.2. Llama-3.2-1B prefill ≈ 890 tok/s.

### A. Resume an idle chat, no pressure (TTFT after a 32-token turn)
| Model | History | KV | Recompute | RAM | SSD file (warm) | Purgeable |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-4bit | 1K | 32 MB | 1.10 s | 0.04 s | 0.04 s | 0.06 s |
| | 8K | 256 MB | 10.11 s | 0.07 s | 0.08 s | 0.10 s |
| | 16K | 512 MB | 24.09 s | 0.09 s | 0.11 s | 0.17 s |
| Llama-3.2-3B-4bit | 1K | 112 MB | 2.85 s | 0.10 s | 0.11 s | 0.12 s |
| | 8K | 896 MB | 28.34 s | 0.17 s | 0.22 s | 0.37 s |

Every tier produced output identical to an uninterrupted run. Purgeable is *slower*
than a warm SSD file because the prototype copies through numpy. A zero-copy C++/MLX
version would roughly match RAM, a gain of ≤0.2 s.

### B. Opening an 8 GB app with 12 idle chats (3 reps each, real distinct text)
| Mode | Swap-out while app opens | App re-read of its 8 GB | Resume all 12 | Chats lost → re-prefilled | SSD written | LLM footprint idle |
|---|---|---|---|---|---|---|
| RAM (status quo) | 4.7 / 5.3 / 3.2 GB | 1.2 / **11.6** / 1.1 s | 5.9 / 7.9 / 9.3 s | 0 | 0 | 3.9–4.5 GB |
| **SSD only** (write at idle, drop RAM) | 0.2 / 0 / 0 GB | 0.9 / 0.9 / 1.1 s | **2.6 / 2.3 / 2.7 s** | 0 | 3.0 GB | 0.84 GB |
| Purgeable + SSD copy at idle | 0 / 0 / 1.0 GB | 0.8 / 0.8 / 0.8 s | 3.2 / 3.3 / 4.8 s | 0 | 3.0 GB | 0.84 GB |
| Purgeable + SSD on pressure warning | 0.1 / 0 / 0 GB | 0.8 / 1.0 / 1.0 s | 129 / 138 / 143 s | **12 / 12 / 12** | 0 | 0.85 GB |
| Purgeable + group-0 "canary" early warning | 1.8 / 0 / 1.7 GB | 1.4 / 2.4 / 0.9 s | 87 / 92 / 95 s | **7 / 7 / 7** | 1.3 GB | 0.85 GB |

All 15 runs matched reference outputs token for token.

**What each row says**
* **SSD only is the winner.** No swap, the fastest resume, and the smallest footprint. Its only cost is writing each idle cache to SSD once.
* **Purgeable + SSD copy** gets the same protection (the kernel discarded 1.8–2.8 GB instead of swapping) but adds nothing over SSD only.
* **The pressure warning comes too late.** `kern.memorystatus_vm_pressure_level` first read "warning" when 0 of 12 caches were left. The kernel purges volatile memory *before* it declares pressure.
* **The canary trick half-works.** Group-0 regions do vanish before group-7 KV, and spilling on that signal saved 5 of 12 chats with 58% fewer SSD writes. But writing GBs to SSD loses the race against an app allocating at ~2 GB/s, and the 7 lost chats cost ~90 s of re-prefill.
* **Group 7 isn't safe at idle either.** Even before the app opened, only 5–12 of 12 caches were still resident. A normally busy Mac purges them on its own.
* **The app slowdown in the RAM baseline varies.** It happened in 1 of 3 reps (11.6 s). In the other two, macOS swapped the LLM's memory instead, which still costs 3–5 GB of SSD swap writes.

### C. Coverage (`research/test_cache_types.py`)
Round trip through the store, then 24 greedy tokens, compared with an uninterrupted run. All four were identical:
Llama `KVCache`, Llama 8-bit `QuantizedKVCache`, Gemma-3 `RotatingKVCache`+`KVCache`,
and Qwen3.5 hybrid `ArraysCache`+`KVCache`.

### D. Kernel facts worth knowing (`research/test_purgeable.py`)
* Volatile pages leave `phys_footprint` at once (a 512 MB object: 1085 → 573 MB).
* Under a low free-page pool, the default group 0 was purged instantly while group 7 survived. With plenty of free memory, neither was purged.

### E. Making the SSD tier write less: lossless compression (`research/bench_compress.py`)
Real 8K-token caches, Apple's built-in libcompression, 8 threads, every result
round-tripped bit-for-bit. "split" = store the high byte (sign+exponent) and low byte
of each 16-bit value as separate planes before compressing.

| Model (KV dtype) | Raw LZ4 | Raw LZFSE | **Split + LZFSE** | Encode | Decode |
|---|---|---|---|---|---|
| Llama-3.2-1B (fp16) | 1.03× | 1.11× | **1.20×** | 1.3 GB/s | 10.9 GB/s |
| Llama-3.2-3B (fp16) | 1.01× | 1.09× | **1.18×** | 1.2 GB/s | 8.4 GB/s |
| Gemma-3-1B (fp16) | 1.01× | 1.08× | **1.17×** | 1.2 GB/s | 8.4 GB/s |
| Qwen3.5-0.8B (bf16) | 1.00× | 1.22× | **1.34×** | 0.8 GB/s | 10.0 GB/s |

That's 15–25% fewer bytes written to and read from SSD, with zero quality loss. Decode
is faster than the SSD, so it shouldn't slow cold resumes. Encode is slower than the
SSD, which is fine because the write happens while the chat is idle. Generic
compression on raw bytes (what an OS swap compressor effectively does) barely helps.
oMLX, LM Studio's mlx-engine and vllm-mlx write their SSD tiers uncompressed; lossless
exponent coding of bf16 KV is known on GPUs (GAAC, SplitZip, ExANS).
End-to-end resume time with compression was not measured.

### F. All SSD improvements combined (`research/bench_combined.py`, `purgekv/segstore.py`)
Llama-3.2-3B, 4 chats × (4K-token history + 8 turns of 256 tokens), saved after every
turn. Each chat resumed with the page cache bypassed (`F_NOCACHE`, a true cold SSD read).

| Policy | SSD written | vs naive | Cold resume (median) | Outputs identical |
|---|---|---|---|---|
| Full rewrite, raw (naive) | 20.2 GB | 1× | 0.44 s | 4/4 |
| Full rewrite + lossless | 17.1 GB | 1.18× | 0.84 s | 4/4 |
| Append-only, raw | 2.7 GB | 7.5× | 0.50 s | 4/4 |
| **Append + lossless** | 2.3 GB | **8.8×** | 0.91 s | 4/4 |
| **Append + 8-bit** (lossy) | 1.4 GB | **14.1×** | **0.40 s** | 4/4 (24/24 tokens) |
| Recompute instead | 0 | — | 19.2 s | — |

* **Append-only is almost the whole win.** It writes each token's KV once instead of on every save.
* **8-bit storage** halves bytes again and resumed fastest (less to read). It matched in 4/4 resumes here, but **in Exp H it changed the greedy output in 3 of 6 short-turn resumes** (15/18 overall). It is lossy, so check quality on your own tasks before relying on it.
* **Lossless compression** saves a further 15% but *slowed* cold resumes (0.9 s against 0.5 s). The prototype's numpy plane re-interleaving is the bottleneck; native code would be needed to make it free.
* Append-only (oMLX blocks, LM Studio 256-token records) and 8-bit (oMLX Affine8) already exist. The lossless part is the only new piece, and it's marginal.

### G. Estimate: large models on a Mac Studio (`research/estimate_studio.py`)
**Not measured.** It applies F's measured ratios to published model shapes, with
assumed hardware: SSD ~5 GB/s on M3 Ultra and ~13 GB/s on the M5 generation, and
long-context prefill ranges from community benchmarks. Workload: an agent session
growing to 32K tokens, saved every 1K tokens, 20 sessions/day.

| Model | KV at 32K | Recompute | Cold resume, 8-bit (M3U / M5) | SSD writes/session: full rewrite → append+8-bit | Per year |
|---|---|---|---|---|---|
| Llama-3.3-70B dense (320 KB/token) | 10.7 GB | 2–5.5 min | 1.1 s / 0.4 s | 177 GB → 5.7 GB | 1,293 TB → 42 TB |
| Qwen3-235B-A22B MoE (188 KB/token) | 6.3 GB | 1–2 min | 0.7 s / 0.3 s | 104 GB → 3.3 GB | 760 TB → 24 TB |
| DeepSeek-V3/R1 671B, MLA (69 KB/token) | 2.3 GB | 2.5–3.5 min | 0.2 s / 0.1 s | 38 GB → 1.2 GB | 277 TB → 9 TB |

What changes at Mac Studio scale:
* **The resume gap grows.** Recompute is minutes, and SSD resume is about a second. The SSD tier matters more on a Studio than on a laptop.
* **SSD wear becomes the real risk.** A naive full-rewrite tier on a 70B agent would write about 1.3 PB a year. Apple doesn't publish endurance ratings, but typical 1–2 TB NVMe drives are rated at roughly 600–1,200 TB written, and Mac SSDs are soldered, so a worn drive means a logic-board repair. Append+8-bit brings that to about 42 TB a year.
* **MLA models (DeepSeek) benefit least**, because their KV cache is already compressed about 5× by design.
* Caveats: the recompute ranges understate long-context cost, since attention grows quadratically. Real SSD throughput through Python was ~1.5–2 GB/s here, below the spec numbers used. No Mac Studio was tested.

### H. Going beyond existing tiers: layer-streamed resume (`purgekv/stream.py`, `research/bench_stream.py`)
oMLX, LM Studio's mlx-engine and vllm-mlx all load a session's whole KV cache from SSD,
*then* run the new tokens. Streaming starts the forward pass at once and reads layer
i+1's KV (on 4 I/O threads) while the GPU computes layer i. With unified memory the
SSD bytes go straight into GPU-visible memory, so there is no host-to-device copy
stage. Same I/O code in both arms; cold SSD reads; 3B model, 6K-token history in
9 append segments; median of 6 resumes.

| New turn | Load then compute | **Streamed** | Speedup | Lossless output |
|---|---|---|---|---|
| 32 tokens | 0.46 s | **0.29 s** | **1.59×** | identical 6/6 |
| 256 tokens | 1.25 s | **0.99 s** | 1.26× | identical 6/6 |
| 1024 tokens | 4.16 s | 4.02 s | 1.04× | identical 6/6 |
| 32 tokens, 8-bit store | 0.40 s | **0.21 s** | **1.86×** | 3/6 same as reference |

* The gain is `(load + compute) / max(load, compute)`: biggest when the two are similar, nothing when one dominates. On this laptop that's short turns, the common case for agent tool results and chat replies.
* **On a Mac Studio (estimate)** a 70B cache takes ~2 s to read, so the sweet spot moves to 256–1,024-token turns, about 1.4–1.7× by the same formula.
* Novelty: layer-wise KV loading is established on NVIDIA servers (LMCache, CachedAttention). I found no Mac/MLX implementation. This is a port that fits Apple Silicon, not a new concept.
* Limits: Llama-style models and the standard KV cache only; one machine.

## Conclusions
1. **For a local LLM server on a Mac: write idle KV caches to SSD and drop them from RAM.**
   This removed the swap storm completely and resumed fastest. It's simple, and it's
   what mlx-lm's `LRUPromptCache` should do for entries that go idle. oMLX-style SSD
   tiers already exist, but they keep a RAM hot tier, and that hot tier is the part
   that swaps.
2. Purgeable memory is a correct, novel and portable-to-MLX mechanism, but its one
   unique benefit (skipping SSD writes) can't be collected safely. The kernel
   purges faster than any process can react, and that's a race, not a code-speed
   issue. A C++ zero-copy version would shave ≤0.2 s off resumes and wouldn't change this.
3. Remaining niche: purgeable KV makes sense only where losing a cache is cheap,
   e.g. short histories (1–2K tokens re-prefill in ~1–3 s) or an SSD-write budget
   that matters more than latency.

## Limits
One machine (M4, 16 GB, on battery) and one macOS version. 3 reps per mode with
system-wide counters. A synthetic memory hog stands in for a real app. I didn't run
oMLX, LM Studio or Ollama directly. The earlier Low Power Mode runs are in
[`results/lpm_on/`](../results/lpm_on); their baseline under-allocated KV and they're superseded.

## Files
```
purgekv/purgeable.py   ctypes: mach_vm_allocate(VM_FLAGS_PURGABLE), mach_vm_purgable_control
purgekv/store.py       PurgeableKVStore (any mlx-lm cache type), spill-to-SSD, thread-safe
purgekv/common.py      prefill/greedy, real-text sessions, vm_stat/swap/footprint/pressure probes
purgekv/hog.py         stand-in foreground app
research/bench_latency.py       A      research/bench_pressure2.py  B (4 modes)   research/bench_pressure3.py  B (canary)
research/test_cache_types.py    C      research/test_purgeable.py   D             results/  raw JSON
research/bench_compress.py      E      research/bench_combined.py   F             research/estimate_studio.py  G
purgekv/stream.py      layer-streamed resume     research/bench_stream.py     H
purgekv/segstore.py    append-only SSD tier with raw / lossless / 8-bit codecs
research/bench_pressure.py      first pressure version (Low Power Mode runs, superseded by v2)
research/run_all.sh    reproduce everything      launch/make_video.py, make_video2.py  render the launch videos (v2: `--no-voice`)
```
Reproduce everything: `./research/run_all.sh` (Apple Silicon, Python ≥ 3.10). MIT licensed.
