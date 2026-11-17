"""Exp I: does streamed resume hold up for large chats? Measured, not estimated.

  python research/bench_scale.py [--model M] [--hist 8192,16384,32768] [--new 32,256]
                                 [--reps 3] [--shared-chats 4]

For each history length it builds a real-text chat (saving every 4096 tokens, as a
long conversation would), then records with cold SSD reads:
  build_s            recomputing the history from scratch (the no-cache alternative)
  load_s[reader]     reading every layer, no compute
  compute_s          the new tokens on a cache already in RAM (no copy, then trimmed back)
  ttft[method/reader] load_then_compute and streamed resume, legacy / single / native readers
  prefetched         streamed resume after a background prefetch finished (user think time)
  peak_gb            MLX peak memory during each resume
  identical          greedy output equals the never-saved cache's
Sizes whose weights + cache would not fit the GPU working set are logged "does not fit".
--shared-chats N forks N chats from one cached prompt (the largest size that fits).
"""
import argparse, json, os, shutil, statistics as st, tempfile, time
import mlx.core as mx
from mlx_lm import load
from purgekv.common import real_text_tokens, greedy
from purgekv.segstore import SegmentKVStore
from purgekv.stream import stream_prefill, load_then_compute, prefetch, _layout, _submit_reads, _fill, check_supported
from mlx_lm.models.cache import make_prompt_cache, KVCache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
ap.add_argument("--hist", default="8192,16384,32768")
ap.add_argument("--new", default="32,256")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--readers", default="legacy,single,native")
ap.add_argument("--shared-chats", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
HIST = [int(x) for x in args.hist.split(",")]
NEW = [int(x) for x in args.new.split(",")]
READERS = args.readers.split(",")
CHECK = 8

mx.set_cache_limit(256 << 20)
model, tok = load(args.model)
check_supported(model)
weights_gb = mx.get_active_memory() / 1e9
budget_gb = mx.device_info()["max_recommended_working_set_size"] / 1e9
root = tempfile.mkdtemp(prefix="purgekv-scale-", dir=os.environ.get("SPILL_DIR"))
GB = 1e9


def per_token_bytes():
    c = make_prompt_cache(model)
    out = model(mx.array(real_text_tokens(tok, 8, 0))[None], cache=c)
    mx.eval(out)
    return sum(x.keys[..., :x.offset, :].nbytes + x.values[..., :x.offset, :].nbytes for x in c) / 8


PER_TOKEN = per_token_bytes()


def build(n, store, key):
    """Prefill n real tokens, saving every 4096 like a long conversation. Returns cache, seconds."""
    cache = make_prompt_cache(model)
    toks = real_text_tokens(tok, n, 1)
    t = time.perf_counter()
    for i in range(0, n, 2048):
        out = model(mx.array(toks[i:i + 2048])[None], cache=cache)
        mx.eval(out, [c.state for c in cache])
        if (i + 2048) % 4096 == 0 or i + 2048 >= n:
            store.save(key, cache)
    return cache, time.perf_counter() - t


def trim(cache, k):
    for c in cache:
        c.trim(k)


def resume(fn, store, key, new, reader):
    mx.clear_cache(); mx.reset_peak_memory()
    t = time.perf_counter()
    c, lg = fn(model, store, key, new, reader=reader)
    mx.eval(mx.argmax(lg, -1))
    dt = time.perf_counter() - t
    peak = mx.get_peak_memory() / GB
    out = greedy(model, c, lg, CHECK)
    del c, lg
    mx.clear_cache()
    return dt, peak, out


def load_only(store, key, reader):
    mx.clear_cache()
    layout = _layout(store, key)
    t = time.perf_counter()
    reader, futs = _submit_reads(store, layout, len(model.model.layers), 0, reader)
    cs = []
    for f in futs:
        c = KVCache(); _fill(c, f.result(), reader, store.codec, layout[0][2]); cs.append(c)
    mx.eval([x for c in cs for x in (c.keys, c.values)])
    dt = time.perf_counter() - t
    del cs
    mx.clear_cache()
    return dt


rows = []
info = {"model": args.model, "weights_gb": round(weights_gb, 2), "gpu_budget_gb": round(budget_gb, 2),
        "kv_bytes_per_token": int(PER_TOKEN), "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
        "ram_gb": int(os.popen("sysctl -n hw.memsize").read()) // 2**30}
print(json.dumps(info), flush=True)
largest_fit = None
for n in HIST:
    kv_gb = n * PER_TOKEN / GB
    need = weights_gb + 2.2 * kv_gb + 0.5  # cache + one transient copy while reading + activations
    row = {"hist": n, "kv_gb": round(kv_gb, 2), "need_gb": round(need, 2)}
    if need > budget_gb:
        row["status"] = "does not fit"
        rows.append(row); print(json.dumps(row), flush=True)
        continue
    store = SegmentKVStore(os.path.join(root, f"h{n}"))
    cache, row["build_s"] = build(n, store, "chat")
    row["segments"] = len(store.sessions["chat"]["segs"])
    refs, row["compute_s"] = {}, {}
    for m in NEW:
        new = real_text_tokens(tok, m, 7 + m)
        ts = []
        for _ in range(args.reps + 1):
            t = time.perf_counter()
            lg = model(mx.array(new)[None], cache=cache)[:, -1, :]
            mx.eval(lg)
            ts.append(time.perf_counter() - t)
            trim(cache, m)
        row["compute_s"][m] = round(st.median(ts[1:]), 3)
        lg = model(mx.array(new)[None], cache=cache)[:, -1, :]
        refs[m] = (new, greedy(model, cache, lg, CHECK))
        trim(cache, m + CHECK)
    del cache
    mx.clear_cache()
    row["load_s"] = {r: round(st.median([load_only(store, "chat", r) for _ in range(args.reps)]), 3) for r in READERS}
    row["ttft"], row["peak_gb"], row["identical"] = {}, {}, {}
    for m in NEW:
        new, ref = refs[m]
        for name, fn in (("load_then_compute", load_then_compute), ("streamed", stream_prefill)):
            for r in READERS:
                resume(fn, store, "chat", new, r)  # warm-up for this shape
                res = [resume(fn, store, "chat", new, r) for _ in range(args.reps)]
                k = f"{name}/{r}/{m}"
                row["ttft"][k] = round(st.median(x[0] for x in res), 3)
                row["peak_gb"][k] = round(max(x[1] for x in res), 2)
                row["identical"][k] = all(x[2] == ref for x in res)
        # prefetch started when the chat was opened; the user typed for longer than the read took
        ts, ok = [], True
        for _ in range(args.reps + 1):
            mx.clear_cache()
            p = prefetch(model, store, "chat", headroom=m)
            while not p.done():
                time.sleep(0.005)
            t = time.perf_counter()
            c, lg = stream_prefill(model, store, "chat", new, prefetched=p)
            mx.eval(mx.argmax(lg, -1))
            ts.append(time.perf_counter() - t)
            ok &= greedy(model, c, lg, CHECK) == ref
            del c, lg, p
        row["ttft"][f"prefetched/single/{m}"] = round(st.median(ts[1:]), 3)
        row["identical"][f"prefetched/single/{m}"] = ok
    base = row["ttft"][f"load_then_compute/legacy/{NEW[0]}"]
    row["speedup_streamed_vs_legacy"] = {f"{r}/{m}": round(row["ttft"][f"load_then_compute/legacy/{m}"] /
                                                           row["ttft"][f"streamed/{r}/{m}"], 2)
                                         for m in NEW for r in READERS}
    row["status"] = "ok"
    rows.append(row); print(json.dumps(row), flush=True)
    largest_fit = (n, store)
    _ = base

shared = None
if args.shared_chats and largest_fit:
    n, _ = largest_fit
    store = SegmentKVStore(os.path.join(root, "shared"))
    prompt, _ = build(n, store, "doc")
    prompt_bytes = sum(os.path.getsize(p) for p, _ in store.sessions["doc"]["segs"])
    shared = {"prompt_tokens": n, "chats": args.shared_chats, "own_tokens": 256, "ttft": [], "identical": []}
    refs = []
    for j in range(args.shared_chats):
        own = real_text_tokens(tok, 256, 500 + j)
        out = model(mx.array(own)[None], cache=prompt); mx.eval(out)
        store.fork("doc", f"chat{j}")
        store.save(f"chat{j}", prompt)  # writes only this chat's 256 tokens
        q = real_text_tokens(tok, 32, 900 + j)
        lg = model(mx.array(q)[None], cache=prompt)[:, -1, :]
        refs.append((q, greedy(model, prompt, lg, CHECK)))
        trim(prompt, 256 + 32 + CHECK)
    del prompt
    mx.clear_cache()
    disk = sum(os.path.getsize(os.path.join(store.root, f)) for f in os.listdir(store.root))
    shared["disk_gb"] = round(disk / GB, 3)
    shared["disk_gb_if_copied"] = round((prompt_bytes * args.shared_chats + (disk - prompt_bytes)) / GB, 3)
    for j, (q, ref) in enumerate(refs):
        dt, peak, out = resume(stream_prefill, store, f"chat{j}", q, "single")
        shared["ttft"].append(round(dt, 3)); shared["identical"].append(out == ref)
    print(json.dumps({"shared": shared}), flush=True)

shutil.rmtree(root, ignore_errors=True)
os.makedirs("results", exist_ok=True)
out = args.out or f"results/scale_{args.model.split('/')[-1]}.json"
json.dump({"info": info, "rows": rows, "shared": shared}, open(out, "w"), indent=1)
print("wrote", out)
