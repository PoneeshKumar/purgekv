# Contributing

Thanks for helping. The most useful contributions right now:

1. **Benchmark numbers from other Macs.** Run `python examples/demo.py` and open a
   [benchmark issue](https://github.com/PoneeshKumar/purgekv/issues/new?template=benchmark.yml).
   M1–M3, Pro/Max/Ultra and Mac Studio results are all missing.
2. **More model families.** `purgekv/stream.py` supports plain decoder stacks with a
   standard `KVCache` (Llama 3.x, Qwen3). Sliding-window (Gemma 3) and hybrid
   linear-attention (Qwen3.5) caches are the next step; see `check_supported()`.
3. **Server integration**, e.g. resuming through `mlx_lm.server`'s prompt cache.

## Setup

```bash
git clone https://github.com/PoneeshKumar/purgekv && cd purgekv
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check .
pytest                          # fast tests, no model download
PURGEKV_MODEL_TESTS=1 pytest    # if you touched stream.py or segstore.py
```

- A change to `stream.py` must keep output **identical** to load-then-compute with
  `codec="raw"`. The model tests check this.
- Performance claims need numbers: chip, RAM, model, context length, and median of
  several runs with cold SSD reads, as `examples/demo.py` prints them.
- Keep PRs small and focused; one change per PR.
