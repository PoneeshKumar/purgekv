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

### I. Large chats: 8K–64K tokens, measured (`research/bench_scale.py`)
M4 MacBook, 16 GB, battery, Low Power Mode off. Real text, saved every 4096 tokens
(as a long conversation would be), cold SSD reads, median of 3. "Today" is
load-then-compute with the legacy reader; streamed and prefetched use the single-copy
reader. Every resume, every reader, every size: output identical to the cache that
never left memory.

**Llama 3.2 3B** (112 KB of KV per token)

| History | KV | Recompute | Today, 32-tok msg | Streamed | **Prefetched** | Today, 256-tok msg | Streamed | Prefetched |
|---|---|---|---|---|---|---|---|---|
| 8K | 0.9 GB | 27 s | 0.54 s | 0.32 s | **0.15 s** | 1.37 s | 1.02 s | 0.99 s |
| 16K | 1.9 GB | 62 s | 0.96 s | 0.63 s | **0.21 s** | 2.02 s | 1.37 s | 1.28 s |
| 32K | 3.8 GB | 169 s | 1.80 s | 1.40 s | **0.31 s** | 3.56 s | 2.08 s | 1.87 s |

**Llama 3.2 1B** (32 KB of KV per token)

| History | KV | Recompute | Today, 32-tok msg | Streamed | **Prefetched** | Today, 256-tok msg | Streamed | Prefetched |
|---|---|---|---|---|---|---|---|---|
| 8K | 0.3 GB | 10 s | 0.16 s | 0.10 s | **0.06 s** | 0.48 s | 0.41 s | 0.39 s |
| 16K | 0.5 GB | 25 s | 0.31 s | 0.19 s | **0.09 s** | 0.73 s | 0.55 s | 0.51 s |
| 32K | 1.1 GB | 69 s | 0.57 s | 0.40 s | **0.12 s** | 1.16 s | 0.82 s | 0.72 s |
| 64K | 2.2 GB | 205 s | 1.06 s | 0.80 s | **0.19 s** | 2.06 s | 1.37 s | 1.17 s |

* **Prefetch is the big one.** Reading the cache while the user types makes a short-message resume compute-only: 4.8–5.8× faster than today at 32–64K.
* **Streaming alone** gives 1.3–1.7× at large sizes. For short messages on big chats the SSD read dominates, so the 32-token gain shrinks (1.3× at 32K); for 256-token messages it grows (1.7×).
* **Single-copy reader:** at 32K (3B) it loaded 21% faster than the legacy reader (1.30 s vs 1.64 s) and lowered peak GPU memory from 7.1 GB to 5.8 GB for load-then-compute. With streaming at 64K (1B), peak fell from 4.4 GB (today) to 3.0 GB.
* **C++ reader:** within 3% of the Python reader everywhere. Python's `preadv` already releases the GIL, so native code buys nothing here.
* **Memory is the real ceiling, not speed.** 3B at 64K (7.3 GB of KV) was not attempted: weights plus cache plus one transient copy would exceed the ~12.7 GB GPU working set.

**Shared cached prompt** (3B): one 32K-token prompt, four chats forked from it with
256 tokens each. Disk: **3.9 GB vs 15.2 GB** if each chat stored its own copy; all four
resumed with identical output (1.3–2.1 s).

**The estimate holds up.** `research/scale_estimate.py --fit` fits SSD speed and a
compute model to these runs; its predictions land within 1–10% of the measured
streamed times. Use it for Macs and sizes not measured here, labelled as an estimate.

**Correction to Exp H:** its "compute-only" baseline (1.21 s for 256 tokens) was slower
than the full streamed resume (0.99 s), which is impossible. That baseline started from a
cache rebuilt at exactly its own size, so the first new token forced a full-layer
reallocation inside the timing. bench_scale.py measures compute on the live cache and
trims it back, with no copy.

### J. Other 16 GB inefficiencies, measured (`research/probe_16gb.py`)
Synthetic 2 GB cache shaped like Llama 3.2 3B, cold reads, median of 3.

| Suspected problem | Measured | Verdict |
|---|---|---|
| purgekv's reader leaves SSD speed unused | 2.57 GB/s (C++: 2.82) vs 3.19 GB/s best plain parallel read (32 MB × 4 threads) | **Real, ~12–20%.** Worth tuning read size and threads. |
| Many small segments slow resume | 64 segments 0.77 s vs 1 segment 0.63 s (+23%); compacted 0.64 s | **Real, fixed** by `store.compact()`. A smaller 0.5 GB run showed 6×, where segments were tiny. |
| save() doubles memory and stalls | +14% time vs a raw write; +0.14 GB process memory | Minor. Not worth fixing now. |
| MLX keeps freed memory after a resume | 0.0 GB held | **Not a problem.** |

### K. A server for agents: SSD tier, prefetch, 8-bit KV (`purgekv/server.py`, `research/bench_server.py`)
Four coding agents share one server (Llama 3.2 3B, 4-bit). Each has its own ~8K-token
context (a different slice of the Python standard library as its "repository"). A turn
sends the agent's history plus a ~150-token tool result over real HTTP, streams a
24-token greedy reply, then the agent "runs a tool" for 1.5 s while the next agent goes.
RAM budget for cached chats: 1.2 GB (`--prompt-cache-bytes`), about one agent's chat.
3 measured rounds × 4 agents, then a server restart and one more turn each.

| Time to first token | stock mlx_lm.server | 1: SSD tier | **1+2: + prefetch** | **1+2+3: + 8-bit KV** |
|---|---|---|---|---|
| median | 29.3 s | 1.41 s | **1.33 s** | **1.13 s** |
| 90th percentile | 30.3 s | 1.53 s | 1.44 s | **1.15 s** |
| worst turn | 33.4 s | 2.78 s | 2.51 s | **1.19 s** |
| first turn after a restart (median) | 45.6 s | 2.19 s | 1.85 s | **1.35 s** |
| chat read from SSD (server log, median) | — | 0.40 s | 0.15 s | **0.06 s** |
| disk used | — | 3.6 GB | 3.6 GB | **1.9 GB** |

- **Stock recomputes.** A chat pushed out of the RAM budget is prefilled again from
  scratch: 22–26× slower than reading it back.
- **Prefetch (2) hit every time.** All 12 steady-state reloads in 1+2 and 1+2+3 found
  the chat already read while other agents worked. It cut the SSD read from 0.40 s to
  0.15 s, but end-to-end time only from 1.41 s to 1.33 s: most of the remaining ~1.1 s
  is the model reading the new tool result, not the disk.
- **8-bit KV (3)** halves every byte moved (save, disk, prefetch) and gave the steadiest
  times (worst turn 1.19 s vs 2.51 s). For reference, stock mlx-lm with `--kv-bits 8`
  kept every chat in RAM (it does not apply `--prompt-cache-bytes` when `--kv-bits` is
  set) and answered in 0.98 s, so 1+2+3 is within 0.15 s of "everything in RAM" while
  using 1.2 GB of RAM for chats.
- **Restart.** Stock mlx-lm starts empty and recomputes every agent (45.6 s, 31.9 s with
  8-bit); purgekv-server reopens the chats from SSD (1.35–2.19 s).
- **Same replies?** 8-bit purgekv vs stock 8-bit: 20/20 identical. Full precision vs
  stock (recompute): 17/20. The control settles where the 3 come from: stock mlx-lm with
  enough RAM to keep every chat (`stock_ram`, its own prompt cache, no purgekv) gave the
  **same replies as purgekv-server in 20/20 turns**, and differs from a full recompute in
  the same 3. Reusing a cached history instead of recomputing it can change a word in
  bf16; purgekv reads back exactly the cache mlx-lm would have kept in RAM.

### L. A real multi-agent task: five trading agents on Qwen3-8B (`examples/trading_agents.py`)
The prompt: *make and backtest a trading strategy, with one agent for ideas, one that
builds the pipeline, a QA agent, a backtester and an agent that executes the trade.*
Five agents share one Qwen3-8B (4-bit) server with a 1 GB RAM budget for their chats, for
4 rounds of idea → code → QA → (fix) → sandboxed backtest → verdict. Each agent keeps its
own conversation, growing to ~5K tokens.

| | stock mlx_lm.server | **purgekv-server** |
|---|---|---|
| Average wait for the first token, per agent turn | 15.7 s | **7.0 s (2.2×)** |
| Turns whose chat had left RAM | recomputed: 21.8 s median (12 turns) | **read from SSD: 6.9 s median (11 turns, 10 prefetched)** |
| Turns with > 2.5K tokens of context | 24.9 s | **7.4 s (3.4×)** |
| Whole workflow, including writing code | 601 s | **434 s** |

How it was measured, and what went wrong first:
- **Fair order.** Each Mac run gets slower the longer the GPU works (turn 1, identical
  work, is ~35% slower in whichever run goes second). Both columns are therefore a first
  run on a cool machine: purgekv from `results/trading_agents/`, stock from
  `results/trading_agents_run1/` (its stock run went first). Stock run second: 17.2 s average.
- **Bug found and fixed.** The first run reused nothing from SSD: Qwen3's chat template
  drops the empty `<think></think>` block from past replies, so a saved chat is never an
  exact prefix of the next request. The SSD tier now reuses the longest shared start and
  trims the rest, as mlx-lm's RAM cache does (`test_rewritten_history_reuses_the_shared_start`).
- The workflows are not token-identical (replies differ slightly between runs, so the
  agents write different code), so compare per-turn averages, not turn by turn.

What the agents produced (synthetic prices, paper only): three rounds of code crashed in
the sandboxed backtest; round 4, equal-weighted momentum, passed with out-of-sample Sharpe
1.09 (train 0.58), max drawdown −22%, no look-ahead. The executor agent's own orders added
up to **9.96× the $100,000 paper account**, so the demo now sizes orders in code and keeps
the agent's text only as a note. Not investment advice.

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
purgekv/server.py      purgekv-server (mlx_lm.server + SSD tier + prefetch)   research/bench_server.py  K
research/bench_pressure.py      first pressure version (Low Power Mode runs, superseded by v2)
research/run_all.sh    reproduce everything      launch/make_video.py, make_video2.py  render the launch videos (v2: `--no-voice`)
```
Reproduce everything: `./research/run_all.sh` (Apple Silicon, Python ≥ 3.10). MIT licensed.
