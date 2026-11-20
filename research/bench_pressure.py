"""Exp B: idle-session KV caches vs. a foreground app that needs the memory.

  python bench_pressure.py MODE SESSIONS TOKENS HOG_GB
  MODE = ram          mlx-lm LRUPromptCache behaviour: idle caches stay as live mx arrays
         purgeable    idle caches live in VOLATILE purgeable VM; purged ones are re-prefilled
         purgeable+ssd  same, but purged ones are reloaded from a safetensors copy
"""
import copy, json, os, subprocess, sys, tempfile, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
from purgekv.common import (fake_prompt, prefill, greedy, vm_counters, swap_used_mb,
                            phys_footprint_mb, Timer)
from purgeable_store import PurgeableKVStore

mode, S, N, HOG = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
MODEL = os.environ.get("MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit")
mx.set_cache_limit(256 << 20)
model, _ = load(MODEL)
pid = os.getpid()
hist, turn = fake_prompt(N, 1), fake_prompt(32, 2)
log = {"mode": mode, "sessions": S, "tokens": N, "hog_gb": HOG, "model": MODEL}

# Build one real history, then S idle sessions holding copies of its KV cache.
base, _ = prefill(model, hist)
ref_c, ref_l = prefill(model, turn, copy.deepcopy(base))
ref = greedy(model, ref_c, ref_l, 16)
del ref_c
log["kv_per_session_mb"] = sum(c.nbytes for c in base) / 2**20

tmp = tempfile.mkdtemp()
store, ram = PurgeableKVStore(), {}
for i in range(S):
    if mode == "ram":
        # distinct buffers, as real sessions would have (deepcopy shares MLX buffers)
        c = copy.deepcopy(base)
        for x in c:
            x.keys, x.values = x.keys + 0, x.values + 0
        mx.eval([x.state for x in c]); ram[i] = c
    else:
        store.put(i, hist, base)
        if mode == "purgeable+ssd":
            save_prompt_cache(os.path.join(tmp, f"{i}.safetensors"), base)
del base
mx.clear_cache()
time.sleep(3)


def snap(tag):
    log[tag] = {"footprint_mb": round(phys_footprint_mb(pid)), "swap_used_mb": swap_used_mb(),
                **vm_counters()}
    if mode != "ram":
        log[tag]["resident_sessions"] = sum(store.survivors().values())


snap("before_hog")
hog = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hog.py"), str(HOG)], stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, text=True)
log["hog_alloc_s"] = json.loads(hog.stdout.readline())["alloc_s"]
snap("after_hog_alloc")


def retouch():
    hog.stdin.write("\n"); hog.stdin.flush()
    return json.loads(hog.stdout.readline())["retouch_s"]


log["hog_retouch_idle_s"] = retouch()  # LLM idle: did the app keep its memory?
snap("after_retouch_idle")

# The user comes back to every idle chat (most recent first) while the app stays open.
ttfts, outcome, correct = [], [], True
for i in reversed(range(S)):
    with Timer() as t:
        if mode == "ram":
            cache, how = copy.deepcopy(ram.pop(i)), "hit"
        else:
            cache, _ = store.get(i)
            how = "hit"
            if cache is None and mode == "purgeable+ssd":
                cache, how = load_prompt_cache(os.path.join(tmp, f"{i}.safetensors")), "ssd"
            elif cache is None:
                cache, _ = prefill(model, hist)
                how = "recompute"
        cache, logits = prefill(model, turn, cache)
        mx.eval(mx.argmax(logits, -1))
    ttfts.append(round(t.s, 3)); outcome.append(how)
    out = greedy(model, cache, logits, 16)
    correct &= out == ref
    del cache
    mx.clear_cache()
log["resume_ttft_s"], log["resume_outcome"], log["outputs_match_reference"] = ttfts, outcome, correct
snap("after_resume")

log["hog_retouch_after_resume_s"] = retouch()
snap("after_retouch_resume")
hog.stdin.close(); hog.wait()

K = ("swapouts_mb", "swapins_mb", "purged_mb")
d = lambda a, b: {k: round(log[b][k] - log[a][k]) for k in K}
log["delta_app_open"] = d("before_hog", "after_retouch_idle")
log["delta_resume"] = d("after_retouch_idle", "after_retouch_resume")
os.makedirs("results", exist_ok=True)
json.dump(log, open(f"results/pressure_{mode}_{S}x{N}_hog{HOG:g}_{os.environ.get('REP', 0)}.json", "w"), indent=1)
summary = {k: log[k] for k in ("mode", "hog_alloc_s", "hog_retouch_idle_s", "hog_retouch_after_resume_s",
                               "outputs_match_reference", "delta_app_open", "delta_resume")}
summary["resume_total_s"] = round(sum(ttfts), 2)
summary["outcomes"] = {o: outcome.count(o) for o in set(outcome)}
summary["footprint_mb"] = [log[t]["footprint_mb"] for t in ("before_hog", "after_hog_alloc")]
summary["resident_after_app_open"] = log["after_hog_alloc"].get("resident_sessions")
print(json.dumps(summary), flush=True)
