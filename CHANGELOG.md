# Changelog

## 0.3.0 — 2026-11-19

A server for agents. Measured with 4 agents sharing one server on an M4 MacBook (16 GB).

### Added
- **`purgekv-server`**: `mlx_lm.server` with an SSD tier under its prompt cache. Every
  mlx_lm.server flag and the OpenAI API work unchanged. Chats pushed out of the RAM budget
  or lost to a restart are read back from SSD instead of recomputed: 29.3 s → 1.33 s per
  agent turn (3B, 8K-token contexts), 45.6 s → 1.85 s after a restart.
- **Think-time prefetch**: chats waiting on their user or a tool are read back from SSD in
  the background when they leave RAM, oldest-waiting first, up to `--prefetch-gb`.
- **8-bit KV on SSD**: mlx-lm `QuantizedKVCache` (`--kv-bits 8`) is stored as is, about
  half the bytes. With prefetch: 1.13 s per agent turn, worst 1.19 s, 1.35 s after a restart.
- `SegmentKVStore.read_raw()`: file I/O only, so reads can run on a background thread.
- `research/bench_server.py` (agents over real HTTP) and `examples/trading_agents.py`
  (five agents design, code, check, backtest and paper-trade a strategy; paper only).


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
