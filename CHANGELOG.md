# Changelog

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
