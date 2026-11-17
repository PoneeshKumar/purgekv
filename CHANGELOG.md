# Changelog

## 0.2.0 — 2026-11-16

Large chats, measured on an M4 MacBook (16 GB) up to 64K tokens.

### Added
- **Prefetch** (`prefetch()` + `stream_prefill(prefetched=...)`): read a chat's KV while
  the user types; resume becomes compute-only. 3B at 32K tokens: 1.80 s → 0.31 s (5.8×).
- **Single-copy reader** (`reader="single"`, now the default for `stream_prefill`): one
  scatter read per segment into a pre-sized buffer. At 32K (3B): 21% faster loads and
  peak GPU memory 7.1 → 5.8 GB. A C++ twin (`reader="native"`) measured no faster.
- **Shared cached prompts:** `store.fork(src, dst)` starts chats from one cached prompt
  without copying it (4 chats on a 32K prompt: 3.9 GB instead of 15.2 GB on disk).
- **Restart-safe index:** `store.save_index()` / `SegmentKVStore.open(root)`.
- **Compaction:** `store.compact(key)` merges a chat's small segments while it is idle.
- Benchmarks: `research/bench_scale.py` (8K–64K), `research/probe_16gb.py`,
  `research/scale_estimate.py` (fitted to measurements; within 1–10% of them).

### Fixed
- Streaming with `codec="lz"` read compressed bytes as if they were raw; it now raises
  a clear error.
- Exp H's compute-only baseline included a full-layer reallocation (see RESEARCH.md).

## 0.1.0 — 2026-11-16

First release.

### Added
- **Layer-streamed resume** (`purgekv.stream.stream_prefill`): reads layer i+1's KV
  cache from SSD while the GPU computes layer i. Resuming a 6K-token chat with a short
  message on an M4 MacBook: 0.46 s → 0.29 s (1.6×), with bit-identical output.
- **Append-only SSD KV tier** (`purgekv.segstore.SegmentKVStore`): writes only new
  tokens on each save, with `raw` (lossless), `lz` (lossless, compressed) and `q8`
  (8-bit, lossy) codecs.
- `check_supported(model)`: Llama 3.x and Qwen3 are supported; other cache layouts are
  refused with a clear error.
- `examples/demo.py`: measure the speedup on your own Mac in one command.
- Test suite (fast, plus opt-in model tests), lint CI, issue templates.
- Research log (`docs/RESEARCH.md`) with all experiments and raw results.

### Fixed
- `codec="q8"` crashed on bf16 models (numpy has no bfloat16).
