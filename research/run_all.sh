#!/bin/zsh
# Reproduce every experiment in README.md on an Apple Silicon Mac (Python >= 3.10).
# Experiment B opens an 8 GB stand-in app and briefly puts the Mac under real memory pressure.
set -e
cd "$(dirname "$0")/.."  # run from the repo root so results/ lands there
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e .
PY=.venv/bin/python

$PY research/test_purgeable.py                                                   # D: kernel semantics
$PY research/test_cache_types.py                                                 # C: all cache types
$PY research/bench_latency.py mlx-community/Llama-3.2-1B-Instruct-4bit 1024,4096,8192,16384   # A
$PY research/bench_latency.py mlx-community/Llama-3.2-3B-Instruct-4bit 1024,4096,8192
for rep in 1 2 3; do                                                    # B: memory pressure
  for m in ram ssd purgeable+ssd purgeable+lazy; do $PY research/bench_pressure2.py $m $rep; sleep 20; done
  $PY research/bench_pressure3.py purgeable+canary $rep; sleep 20
done
$PY research/bench_compress.py mlx-community/Llama-3.2-1B-Instruct-4bit mlx-community/Qwen3.5-0.8B-4bit   # E
$PY research/bench_combined.py                                                   # F: combined SSD tier
$PY research/bench_stream.py                                            # H
$PY research/probe_16gb.py                                              # J: 16 GB inefficiencies
$PY research/bench_scale.py --shared-chats 4                            # I: 3B, 8K-32K
$PY research/bench_scale.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --hist 8192,16384,32768,65536   # I: 1B to 64K
#                                                     # H: layer-streamed resume
python3 research/estimate_studio.py                                              # G: Mac Studio estimate
