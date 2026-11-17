"""Exp J: where does a 16 GB MacBook lose time or memory around the KV cache tier?

  python research/probe_16gb.py [--gb 2]

Synthetic KV shaped like Llama 3.2 3B (28 layers, 8 KV heads, head dim 128, bf16),
so no model download is needed. Probes:
  ssd        raw read throughput: purgekv's per-layer scatter reads vs plain parallel
             reads at several chunk sizes and thread counts (cold, F_NOCACHE)
  segments   the same history stored as 1, 8 and 64 segments: does resume slow down?
  save       store.save() time and the extra process memory its staging copy needs
  retained   MLX memory still held after a resume with mlx-lm's default cache limit
"""
import argparse, fcntl, json, os, resource, shutil, statistics as st, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from purgekv import fastread
from purgekv.segstore import SegmentKVStore, _write
from purgekv.stream import _layout

ap = argparse.ArgumentParser()
ap.add_argument("--gb", type=float, default=2.0, help="size of the synthetic cache")
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
L, H, D = 28, 8, 128
PER_TOKEN = L * 2 * H * D * 2
TOKENS = int(args.gb * 1e9 / PER_TOKEN) // 256 * 256
root = tempfile.mkdtemp(prefix="purgekv-probe-", dir=os.environ.get("SPILL_DIR"))
out = {"tokens": TOKENS, "cache_gb": round(TOKENS * PER_TOKEN / 1e9, 2)}
print(json.dumps(out), flush=True)


def synthetic_cache(n, seed=0):
    cache = []
    for i in range(L):
        k = mx.random.normal((1, H, n, D), key=mx.random.key(seed + i)).astype(mx.bfloat16)
        c = KVCache(); c.update_and_fetch(k, k * 0.5); cache.append(c)
    mx.eval([c.state for c in cache])
    return cache


def store_with_segments(n_segs, cache, name):
    s = SegmentKVStore(os.path.join(root, name))
    sizes = np.full(n_segs, TOKENS // n_segs); sizes[-1] += TOKENS - sizes.sum()
    t = 0
    for sz in sizes:
        t += int(sz)
        view = []
        for c in cache:
            v = KVCache(); v.keys, v.values, v.offset = c.keys, c.values, t; view.append(v)
        s.save("c", view)
    return s


def timed_load(s, backend):
    layout = _layout(s, "c")
    t = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: fastread.read_layer(layout, i, 0, backend), range(L)))
    return time.perf_counter() - t


# ---- ssd: purgekv reader vs plain parallel reads ----
cache = synthetic_cache(TOKENS)
s1 = store_with_segments(1, cache, "seg1")
path = s1.sessions["c"]["segs"][0][0]
size = os.path.getsize(path)


def plain_read(chunk, threads):
    def work(off):
        fd = os.open(path, os.O_RDONLY); fcntl.fcntl(fd, 48, 1)
        buf = bytearray(min(chunk, size - off))
        try:
            os.preadv(fd, [buf], off)
        finally:
            os.close(fd)
    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(work, range(0, size, chunk)))
    return size / (time.perf_counter() - t) / 1e9


out["ssd_gbps"] = {"purgekv_single_reader": round(size / st.median(timed_load(s1, "single") for _ in range(args.reps)) / 1e9, 2),
                   "purgekv_native_reader": round(size / st.median(timed_load(s1, "native") for _ in range(args.reps)) / 1e9, 2)}
for chunk_mb in (1, 8, 32):
    for threads in (1, 2, 4, 8, 16):
        out["ssd_gbps"][f"plain_{chunk_mb}MB_x{threads}"] = round(st.median(plain_read(chunk_mb << 20, threads) for _ in range(args.reps)), 2)
print(json.dumps({"ssd_gbps": out["ssd_gbps"]}), flush=True)

# ---- segments: 1 vs 8 vs 64 ----
out["segments_load_s"] = {}
for n in (1, 8, 64):
    s = s1 if n == 1 else store_with_segments(n, cache, f"seg{n}")
    out["segments_load_s"][n] = round(st.median(timed_load(s, "single") for _ in range(args.reps)), 3)
t = time.perf_counter()
s.compact("c")  # the 64-segment store
out["segments_load_s"]["64_then_compacted"] = round(st.median(timed_load(s, "single") for _ in range(args.reps)), 3)
out["compact_seconds"] = round(time.perf_counter() - t, 3)
print(json.dumps({"segments_load_s": out["segments_load_s"]}), flush=True)

# ---- save: time and peak memory of one big append ----
del cache
mx.clear_cache()
cache = synthetic_cache(TOKENS, seed=100)
s = SegmentKVStore(os.path.join(root, "save"))
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS; a high-water mark
t = time.perf_counter()
s.save("c", cache)
dt = time.perf_counter() - t
out["save"] = {"seconds": round(dt, 3), "write_gbps": round(s.bytes_written / dt / 1e9, 2),
               "process_peak_rise_gb": round((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1e9, 2)}
# the same bytes written with no staging copy, for comparison
arrs = [np.asarray(x.view(mx.uint16)).reshape(-1).view(np.uint8) for c in cache for x in (c.keys, c.values)]
t = time.perf_counter()
_write(os.path.join(root, "direct.bin"), arrs)
out["save"]["direct_write_seconds"] = round(time.perf_counter() - t, 3)
print(json.dumps({"save": out["save"]}), flush=True)

# ---- retained: memory MLX keeps after a resume with no cache limit ----
del cache, arrs
mx.set_cache_limit(mx.device_info()["max_recommended_working_set_size"])  # mlx default
mx.clear_cache()
layout = _layout(s1, "c")
before = mx.get_active_memory() + mx.get_cache_memory()
cs = []
for i in range(L):
    kb, vb, n = fastread.read_layer(layout, i, 32, "single")
    c = KVCache(); c.keys, c.values, c.offset = mx.array(kb).view(mx.bfloat16), mx.array(vb).view(mx.bfloat16), n
    cs.append(c)
mx.eval([c.state for c in cs])
del cs
out["retained"] = {"cache_memory_gb_after_resume": round(mx.get_cache_memory() / 1e9, 2),
                   "baseline_gb": round(before / 1e9, 2)}
mx.clear_cache()
out["retained"]["after_clear_cache_gb"] = round(mx.get_cache_memory() / 1e9, 2)
print(json.dumps({"retained": out["retained"]}), flush=True)

shutil.rmtree(root, ignore_errors=True)
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/probe_16gb.json", "w"), indent=1)
print("wrote results/probe_16gb.json")
