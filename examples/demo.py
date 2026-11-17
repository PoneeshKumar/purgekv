"""See layer-streamed resume on your own Mac, in one command.

  pip install git+https://github.com/PoneeshKumar/purgekv
  python examples/demo.py                      # Llama 3.2 3B, 6K-token chat
  python examples/demo.py --model mlx-community/Qwen3-0.6B-4bit --tokens 4096

Builds a chat, saves its KV cache to SSD (append-only, page cache bypassed), then
resumes it with a short new message two ways and prints both times.
"""
import argparse, copy, platform, shutil, statistics, subprocess, tempfile, time
import mlx.core as mx
from mlx_lm import load
from purgekv.common import real_text_tokens, prefill, greedy
from purgekv.segstore import SegmentKVStore
from purgekv.stream import stream_prefill, load_then_compute, check_supported, prefetch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
ap.add_argument("--tokens", type=int, default=6144, help="chat history length")
ap.add_argument("--new", type=int, default=32, help="new message length")
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
print(f"{chip} · {ram} GB · macOS {platform.mac_ver()[0]} · {args.model}")

mx.set_cache_limit(256 << 20)
model, tok = load(args.model)
check_supported(model)
print(f"building a {args.tokens}-token chat…", flush=True)
cache, _ = prefill(model, real_text_tokens(tok, args.tokens, 1))
store = SegmentKVStore(tempfile.mkdtemp(prefix="purgekv-demo-"), codec="raw")
store.save("chat", cache)
new = real_text_tokens(tok, args.new, 2)
ref_cache, logits = prefill(model, new, copy.deepcopy(cache))
reference = greedy(model, ref_cache, logits, 16)
del cache, ref_cache
mx.clear_cache()


def run(fn):
    times, same = [], True
    for i in range(args.reps + 1):  # first run warms up kernels
        t = time.perf_counter()
        c, lg = fn(model, store, "chat", new)
        mx.eval(mx.argmax(lg, -1))
        dt = time.perf_counter() - t
        same &= greedy(model, c, lg, 16) == reference
        if i:
            times.append(dt)
        del c
        mx.clear_cache()
    return statistics.median(times), same


def prefetched(model, store, key, new):
    p = prefetch(model, store, key, headroom=len(new))  # e.g. when the chat is opened
    while not p.done():  # ...and the user is still typing
        time.sleep(0.005)
    return p


def run_prefetched():
    times = []
    for i in range(args.reps + 1):
        p = prefetched(model, store, "chat", new)
        t = time.perf_counter()
        c, lg = stream_prefill(model, store, "chat", new, prefetched=p)
        mx.eval(mx.argmax(lg, -1))
        if i:
            times.append(time.perf_counter() - t)
        del c, p
        mx.clear_cache()
    return statistics.median(times)


base, base_ok = run(load_then_compute)
fast, fast_ok = run(stream_prefill)
pre = run_prefetched()
shutil.rmtree(store.root, ignore_errors=True)
print(f"\nresume a {args.tokens}-token chat with a {args.new}-token message (cold SSD, median of {args.reps}):")
print(f"  load, then compute : {base:.3f} s")
print(f"  layer-streamed     : {fast:.3f} s   → {base / fast:.2f}× faster")
print(f"  prefetched (cache read while the user types): {pre:.3f} s   → {base / pre:.2f}× faster")
print(f"  output identical to an uninterrupted run: {base_ok and fast_ok}")
print("\nShare your numbers: https://github.com/PoneeshKumar/purgekv/issues/new?template=benchmark.yml")
