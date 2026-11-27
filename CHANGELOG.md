# Changelog

## Unreleased

### Fixed
- **purgekv-server read a whole saved chat to reuse a few tokens.** A new chat that shared
  only the chat template's header (~20 tokens) with a saved chat loaded all of it from SSD
  (six 4K-token agents on Llama 3.2 1B: 5 loads, 970 MB read). A chat is now read back only
  if it saves at least 64 tokens of compute and at least 1/16 of the tokens it reads
  (`MIN_REUSE`, `READ_RATIO`): the same run reads nothing. Agent turns are unchanged
  (4 agents, Llama 3.2 1B: 4.46 s stock → 0.42 s).
- A chat that matches only partway is read up to the segment it needs, not to its end.
- A short match no longer took the saved chat off the prefetch list.

## 0.4.1 — 2026-11-27

### Fixed
- `purgekv.stream` failed to import with mlx-lm main: its model refactor (ml-explore/mlx-lm#2082)
  moved `create_attention_mask` to `mlx_lm.layers.attention`. purgekv now imports it from
  either place, and the full test suite (model tests included) passes on both mlx-lm 0.32.2
  and main (6fe0ce4).
- Server benchmark on mlx-lm main (4 agents, 4K-token chats, 0.3 GB of RAM), median time to
  first token, stock → purgekv-server: Llama 3.2 1B 3.40 s → 0.40 s, Qwen3.5 0.8B
  2.45 s → 0.61 s; after a restart Gemma 3 1B 4.33 s → 0.28 s. Raw data:
  `results/server_mlxlm-main_*.json`.

## 0.4.0 — 2026-11-27

### Added
- **On PyPI:** `pip install purgekv`. Releases are published from GitHub by
  `.github/workflows/release.yml` (PyPI trusted publishing).
- **Disk limit:** `--disk-gb` (default 20, `0` for no limit). When the SSD tier is over the
  limit, the chats used least recently are deleted first; files that chats share are
  counted once and deleted only with the last chat that uses them. The newest chat is
  always kept. The limit also applies to chats left on disk by an earlier run, when the
  server loads that model. `stats.json` reports `disk_bytes` and `evicted`.
- **purgekv-server works with any model mlx-lm runs.** Full-attention layers keep the
  append-only segments; every other layer type (sliding window, linear attention /
  recurrent state, unknown types) is saved whole with mlx-lm's `save_prompt_cache`.
  Chats that cannot be rewound are reused only at their exact saved length, and their
  prompt checkpoints are kept. 4 agents, 8K-token chats, RAM for about one:
  Gemma 3 1B 17.3 s → 1.99 s per turn, Qwen3.5 0.8B 12.3 s → 0.94 s; after a restart
  19.6 s → 1.30 s and 13.8 s → 0.97 s. Replies matched stock mlx-lm with RAM for every
  chat in 20 of 20 turns on both.
- `research/bench_server.py` turns thinking off (`enable_thinking: false`), so models
  that think by default (Qwen3.5) are compared on their replies.

## 0.3.1 — 2026-11-22

### Fixed
- **purgekv-server wrote Qwen3 chats again in full on every turn.** Qwen3's chat template
  drops the empty `<think></think>` from past replies, so a saved chat was never an exact
  prefix of the next one, and every save started from scratch; the old copies also stayed
  on disk. Saves now keep the saved segments before the point where the chats diverge and
  rewrite only the previous turn. In a 10-turn agent chat: 43 MB written → 12.8 MB (6.2 MB
  is the minimum), and disk use 43 MB → 6.3 MB. A chat that branches off before its last
  turn (an edited earlier message) still keeps the original.
- `SegmentKVStore.fork(src, dst, upto=)` shares only the first whole segments;
  `SegmentKVStore.save(key, cache, upto=)` saves up to a token.

### Changed
- The package now holds only the product: `segstore`, `stream`, `fastread`, `server` and
  `common`. The purgeable-memory experiment (`purgekv.purgeable`, `purgekv.store`,
  `purgekv.hog`) moved to `research/` (`purgeable.py`, `purgeable_store.py`, `hog.py`).
  Code that imported `purgekv.store.PurgeableKVStore` must use the research copy.
- Launch videos and their render scripts moved to the `launch` branch; the
  `video` extra is gone.
- README starts with install and `purgekv-server`.

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
  On Qwen3-8B: average wait for the first token 15.7 s → 7.0 s.

### Fixed
- purgekv-server reused nothing from SSD when the chat template rewrites past turns
  (Qwen3 drops empty `<think></think>` blocks). It now reuses the longest shared start
  and trims the rest, like mlx-lm's RAM cache.


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
