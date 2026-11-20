# purgekv

[![lint](https://github.com/PoneeshKumar/purgekv/actions/workflows/lint.yml/badge.svg)](https://github.com/PoneeshKumar/purgekv/actions/workflows/lint.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)
![MLX](https://img.shields.io/badge/built%20on-MLX-orange)

**Keep local LLM agents fast on a Mac that can't hold all their chats in RAM.**
`purgekv-server` is `mlx_lm.server` with an SSD tier under its prompt cache. When a
chat is pushed out of RAM or lost to a restart, its KV cache is read back from SSD in
about a second instead of being recomputed, which takes 20–45 s for an 8K-token chat.

## Install

Needs an Apple Silicon Mac and Python 3.10+.

```bash
pip install git+https://github.com/PoneeshKumar/purgekv
purgekv-server --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt-cache-bytes 2GB
```

It takes every `mlx_lm.server` flag and serves the same OpenAI-compatible API at
`http://localhost:8080/v1`, so chat apps and agent frameworks work unchanged: point
them at that URL. Add `--kv-bits 8` to halve the bytes stored and read (replies can
differ slightly from full precision). Saved chats live in `~/.cache/purgekv/server`
(`--kv-dir` to change it).

## Results

Four agents sharing one server on a 16 GB M4 (8K-token contexts, RAM for about one):

| Time to first token | Stock mlx_lm.server | **purgekv-server** | **+ `--kv-bits 8`** |
|---|---|---|---|
| Typical agent turn | 29.3 s (recomputes the chat) | **1.33 s** | **1.13 s (26×)** |
| Worst turn | 33.4 s | 2.51 s | **1.19 s** |
| After a server restart | 45.6 s | 1.85 s | **1.35 s** |

- After every reply, the chat is appended to SSD (only new tokens). A chat pushed out
  of RAM, or lost to a restart, is read back instead of recomputed.
- **Think-time prefetch:** while an agent runs a tool, its chat is read back from SSD in
  the background, so its next turn does not wait for the disk.
- `--kv-bits 8` (an mlx-lm flag) caches are stored 8-bit: half the bytes on disk and on
  every reload. Replies can differ slightly from full precision (mlx-lm's trade-off).
- Replies match stock mlx-lm with enough RAM to keep every chat: 20 of 20 turns.

**A real multi-agent task.** Five agents (ideas, code, QA, backtest, paper execution)
building a trading strategy on Qwen3-8B, 1 GB of RAM for their chats: average wait for
the first token **15.7 s → 7.0 s**; agents whose chat had left RAM **21.8 s → 6.9 s**.
Run it yourself: `python examples/trading_agents.py` (paper trading only).

Details: [docs/RESEARCH.md](docs/RESEARCH.md#k-a-server-for-agents-ssd-tier-prefetch-8-bit-kv-purgekvserverpy-researchbench_serverpy).

## Library: layer-streamed resume

For your own MLX code, purgekv also streams a saved KV cache
from SSD layer by layer, so the GPU starts computing immediately instead of waiting
for the whole cache to load.

<p align="center"><img src="docs/assets/streaming.gif" alt="SSD reads layer i+1 while the GPU computes layer i; resume drops from 0.46 s to 0.29 s" width="800"></p>

| Resume a 6K-token chat from SSD (M4 MacBook, Llama 3.2 3B) | Load, then compute | **Streamed** | |
|---|---|---|---|
| 32-token new message | 0.46 s | **0.29 s** | **1.6× faster** |
| 256-token new message | 1.25 s | **0.99 s** | 1.3× faster |
| Output | | bit-identical | every resume |

### Large chats, and prefetch

Start reading the cache when the user opens the chat or starts typing, and the resume
is compute-only. Measured on the same M4 MacBook (16 GB), Llama 3.2 3B, 32-token message:

| Chat history | Today: load, then compute | Streamed | **Prefetched** | Recompute instead |
|---|---|---|---|---|
| 8K tokens | 0.54 s | 0.32 s | **0.15 s** | 27 s |
| 16K tokens | 0.96 s | 0.63 s | **0.21 s** | 62 s |
| 32K tokens | 1.80 s | 1.40 s | **0.31 s (5.8×)** | 169 s |

Llama 3.2 1B at **64K tokens**: 1.06 s → 0.80 s streamed → **0.19 s prefetched**, against
205 s to recompute. Output was identical to a never-saved cache in every run.
Details, 256-token messages and memory: [docs/RESEARCH.md](docs/RESEARCH.md#i-large-chats-8k64k-tokens-measured-researchbench_scalepy).

### Try it

```bash
git clone https://github.com/PoneeshKumar/purgekv && cd purgekv
pip install -e .
python examples/demo.py
```

The demo builds a 6K-token chat, saves its KV cache to SSD, and resumes it three ways
with cold reads: load-then-compute, streamed, and prefetched. It prints the times and
checks that the output is identical. Bigger chats: `python research/bench_scale.py`.
Got a different Mac? [Share your numbers](https://github.com/PoneeshKumar/purgekv/issues/new?template=benchmark.yml).

### Use it

```python
from mlx_lm import load
from purgekv.segstore import SegmentKVStore
from purgekv.stream import stream_prefill, prefetch

model, tok = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
store = SegmentKVStore("/path/to/kv-cache", codec="raw")   # append-only SSD tier

store.save("chat-1", cache)              # after each turn: writes only the new tokens
store.compact("chat-1")                  # optional, while idle: merge small segments
store.save_index()                       # survive a restart: SegmentKVStore.open(root)

p = prefetch(model, store, "chat-1")     # when the chat is opened / the user types
cache, logits = stream_prefill(model, store, "chat-1", new_tokens, prefetched=p)
```

**Shared prompts:** cache a long system prompt or document once, then
`store.fork("doc", "chat-7")` starts a chat from it without copying. Four chats on one
32K-token prompt used 3.9 GB of disk instead of 15.2 GB.

`codec="raw"` is lossless. `codec="q8"` halves SSD space and reads, but it is lossy:
it changed the greedy output in 3 of 6 short resumes in our tests.

## How streaming works

<img src="docs/assets/timeline.png" alt="Timeline: load then compute takes 0.46 s; streamed takes 0.29 s" width="800">

Mac LLM servers (oMLX, LM Studio's mlx-engine, vllm-mlx) resume a chat in two steps:
load the entire cache from SSD, then run the new message through the model. Time to
first token is **load + compute**.

But layer *i* only needs layer *i*'s cache, and only when the forward pass reaches
it. `stream_prefill` starts the forward pass at once. Four I/O threads read each
layer's keys and values with the page cache bypassed. The main thread hands each
layer to the GPU with `mx.async_eval`, so the GPU computes layer *i* while layer
*i + 1* is still arriving. Time to first token becomes about **max(load, compute)**.

On Apple Silicon, unified memory means the bytes read from SSD land in memory the
GPU reads directly, with no host-to-device copy to schedule. The gain is largest when
load and compute take similar time: short messages, like agent tool results and quick
replies. Long prompts are compute-bound, so the gain fades.

Layer-wise KV loading is established on NVIDIA servers (LMCache). purgekv brings it
to MLX on the Mac.

## Supported models

| | Status |
|---|---|
| Llama 3.x, Qwen3 | Supported; tested for identical output; Qwen3-8B tested with purgekv-server |
| Gemma 3 (sliding window), Qwen3.5 (hybrid attention) | Not yet supported: purgekv-server keeps them in RAM only (stock behavior); the streaming library refuses them with a clear error |

`check_supported(model)` tells you before you start.

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # fast, no model download
PURGEKV_MODEL_TESTS=1 pytest             # also runs streaming on real models
```

## The research behind it

purgekv started as an investigation into LLM memory on the Mac. Some findings:

- Keeping idle chats' KV caches in RAM pushed 3–5 GB to swap when a big app opened.
- An SSD tier that writes only new tokens cut SSD writes 7.5× against rewriting the whole cache.
- Letting the macOS kernel purge idle caches (purgeable memory) does **not** work: it deletes them before it ever signals memory pressure.

The full log, with every experiment, negative result and raw data, is in
[docs/RESEARCH.md](docs/RESEARCH.md).

## Limits

Tested on one M4 MacBook (16 GB) with the standard KV cache; Llama 3.x and Qwen3 only.
purgekv-server keeps other cache layouts (sliding window, hybrid, draft models) in RAM only,
and does not yet cap its disk use: delete `--kv-dir` (default `~/.cache/purgekv/server`) to clear it.
With chat templates that rewrite earlier replies (Qwen3 drops the empty `<think></think>`),
each save also rewrites the previous turn, so SSD writes are about 2× the minimum.

## License

MIT
