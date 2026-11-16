"""Exp A: cost to resume an idle session (no memory pressure), and correctness.

Resume = get the cache for an N-token history, prefill M new user tokens,
produce the first token. Compared tiers:
  recompute  : no cache, re-prefill N+M tokens
  ram-lru    : mlx-lm LRUPromptCache style (deepcopy of live mx arrays)
  ssd        : save/load_prompt_cache safetensors (oMLX-style SSD tier; OS page cache warm = best case)
  purgeable  : this work
"""
import copy, json, os, sys, tempfile, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
from purgekv.common import fake_prompt, prefill, greedy, Timer
from purgekv.store import PurgeableKVStore

model_id = sys.argv[1]
Ns = [int(x) for x in sys.argv[2].split(",")]
M, CHECK, REPS = 32, 24, 3
mx.set_cache_limit(256 << 20)  # stop the MLX allocator hoarding freed buffers
model, _ = load(model_id)
prefill(model, fake_prompt(512, 0))  # warm up kernels

rows = []
store = PurgeableKVStore()
tmp = tempfile.mkdtemp()
for N in Ns:
    hist, turn = fake_prompt(N, 1), fake_prompt(M, 2)
    base, _ = prefill(model, hist)
    nbytes = sum(c.nbytes for c in base)
    ref_cache, ref_logits = prefill(model, hist + turn)
    ref = greedy(model, ref_cache, ref_logits, CHECK)
    del ref_cache

    def resume(get_cache, toks=turn):
        with Timer() as t:
            cache = get_cache()
            if cache is not None:
                mx.eval([c.state for c in cache])
            acq = time.perf_counter() - t.t
            cache, logits = prefill(model, toks, cache)
            mx.eval(mx.argmax(logits, -1))
        return t.s, acq, greedy(model, cache, logits, CHECK)

    res = {}
    res["recompute"] = [resume(lambda: None, hist + turn)]  # deterministic and slow: 1 rep
    res["ram-lru"] = [resume(lambda: copy.deepcopy(base)) for _ in range(REPS)]
    f = os.path.join(tmp, f"{N}.safetensors")
    with Timer() as ts:
        save_prompt_cache(f, base)
    res["ssd"] = [resume(lambda: load_prompt_cache(f)) for _ in range(REPS)]
    put_s = []
    def pget():
        with Timer() as tp:
            store.put("s", hist, base)
        put_s.append(tp.s)
        c, _ = store.get("s")
        assert c is not None, "purged with no pressure"
        return c
    res["purgeable"] = [resume(pget) for _ in range(REPS)]

    row = {"N": N, "kv_mb": round(nbytes / 2**20, 1), "ssd_save_s": round(ts.s, 3),
           "purgeable_put_s": round(min(put_s), 3)}
    for k, v in res.items():
        row[k + "_ttft_s"] = round(min(t for t, _, _ in v), 4)
        row[k + "_acquire_s"] = round(min(a for _, a, _ in v), 4)
        row[k + "_correct"] = all(o == ref for _, _, o in v)
    base = None  # release this N's cache before the next size
    os.remove(f); mx.clear_cache()
    rows.append(row)
    print(json.dumps(row), flush=True)

out = f"results/latency_{model_id.split('/')[-1]}.json"
os.makedirs("results", exist_ok=True)
json.dump(rows, open(out, "w"), indent=1)
