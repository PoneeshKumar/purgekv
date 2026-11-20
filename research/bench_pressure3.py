"""Exp B v2: 12 distinct idle chats (real text) vs a foreground app that needs the memory.

  python bench_pressure2.py MODE [REP]
  ram            status quo (mlx-lm LRUPromptCache): idle caches stay as live MLX arrays
  ssd            no RAM tier at all: write cache to SSD at idle, drop it, reload on resume
  purgeable+ssd  purgeable hot tier + eager SSD copy at idle
  purgeable+lazy purgeable hot tier; SSD copy written only when macOS reports memory
                 pressure; if the kernel purged first, re-prefill
"""
import json, os, shutil, sys, subprocess, tempfile, threading, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
from purgekv.common import (real_text_tokens, prefill, greedy, vm_counters, swap_used_mb,
                            phys_footprint_mb, pressure_level, Timer)
from purgeable_store import PurgeableKVStore

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
S, N, HOG = 12, 8192, 8
SESS = os.path.join(os.environ.get("SESS_DIR", "sessions"), f"{S}x{N}")
mode, rep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
mx.set_cache_limit(256 << 20)
model, tok = load(MODEL)

# One-time: build 12 distinct real-text sessions + reference continuations.
if not os.path.exists(os.path.join(SESS, "refs.json")):
    os.makedirs(SESS, exist_ok=True)
    refs = {}
    for i in range(S):
        hist, turn = real_text_tokens(tok, N, i), real_text_tokens(tok, 32, 100 + i)
        c, _ = prefill(model, hist)
        save_prompt_cache(os.path.join(SESS, f"{i}.safetensors"), c)
        c, logits = prefill(model, turn, c)
        refs[i] = {"hist": hist, "turn": turn, "ref": greedy(model, c, logits, 16)}
        del c; mx.clear_cache()
    json.dump(refs, open(os.path.join(SESS, "refs.json"), "w"))
refs = {int(k): v for k, v in json.load(open(os.path.join(SESS, "refs.json"))).items()}

pid, tmp = os.getpid(), tempfile.mkdtemp(dir=os.environ.get("SPILL_DIR"))
log = {"mode": mode, "rep": rep, "sessions": S, "tokens": N, "hog_gb": HOG, "model": MODEL}
store, ram, ssd_written = PurgeableKVStore(), {}, 0

# Sessions become idle.
with Timer() as t_idle:
    for i in range(S):
        c = load_prompt_cache(os.path.join(SESS, f"{i}.safetensors"))
        mx.eval([x.state for x in c])
        if mode == "ram":
            ram[i] = c
        elif mode == "ssd":
            f = os.path.join(tmp, f"{i}.safetensors"); save_prompt_cache(f, c)
            ssd_written += os.path.getsize(f)
        else:
            store.put(i, refs[i]["hist"], c)
            if mode == "purgeable+ssd":
                ssd_written += store.spill(i, os.path.join(tmp, f"{i}.bin"))
        del c
mx.clear_cache()
log["stash_all_s"] = round(t_idle.s, 2)
log["ssd_written_at_idle_mb"] = round(ssd_written / 2**20)

# Lazy mode: spill resident entries to SSD as soon as macOS reports pressure.
events, stop = [], threading.Event()
def watcher():
    global ssd_written
    while not stop.is_set():
        lvl = pressure_level()
        if lvl >= 2:
            for i in range(S):
                if store.resident(i):
                    ssd_written += store.spill(i, os.path.join(tmp, f"{i}.bin"))
            events.append((round(time.time(), 2), lvl, sum(store.resident(i) for i in range(S)), len(store.spills)))
        time.sleep(0.02)
from purgeable import PurgeableRegion, VOLATILE
import numpy as np


def canary_watcher():
    """The kernel purges volatile group 0 before group 7. Small group-0 canaries
    vanish first, which is an early warning to spill group-7 KV to SSD."""
    global ssd_written
    def arm():
        r = PurgeableRegion(16 << 20)
        np.frombuffer(r.buffer(), dtype=np.uint8)[:] = 1
        r._set(VOLATILE)  # group 0
        return r
    canaries = [arm() for _ in range(4)]
    while not stop.is_set():
        dead = [c for c in canaries if (c.state() & 3) == 2]
        if dead:
            n0 = len(store.spills)
            for i in range(S):
                if store.resident(i):
                    ssd_written += store.spill(i, os.path.join(tmp, f"{i}.bin"))
            events.append((round(time.time(), 2), "canary", sum(store.resident(i) for i in range(S)),
                           len(store.spills) - n0))
            for c in dead:
                c.free(); canaries.remove(c); canaries.append(arm())
        time.sleep(0.005)


if mode == "purgeable+lazy":
    threading.Thread(target=watcher, daemon=True).start()
if mode == "purgeable+canary":
    threading.Thread(target=canary_watcher, daemon=True).start()
time.sleep(3)


def snap(tag):
    log[tag] = {"footprint_mb": round(phys_footprint_mb(pid)), "swap_used_mb": swap_used_mb(),
                "pressure": pressure_level(), **vm_counters()}
    if mode.startswith("purgeable"):
        log[tag]["resident"] = sum(store.resident(i) for i in range(S))
        log[tag]["spilled"] = len(store.spills)


snap("before_app")
hog = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hog.py"), str(HOG)], stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, text=True)
log["app_alloc_s"] = json.loads(hog.stdout.readline())["alloc_s"]
snap("after_app_alloc")


def retouch():
    hog.stdin.write("\n"); hog.stdin.flush()
    return json.loads(hog.stdout.readline())["retouch_s"]


log["app_retouch_idle_s"] = retouch()
snap("after_retouch_idle")

ttfts, how_all, correct = [], [], []
for i in reversed(range(S)):
    r = refs[i]
    with Timer() as t:
        if mode == "ram":
            cache, how = ram.pop(i), "ram"
        elif mode == "ssd":
            cache, how = load_prompt_cache(os.path.join(tmp, f"{i}.safetensors")), "ssd"
        else:
            hits = store.stats["hit"]
            cache, _ = store.get(i)
            how = "hit" if store.stats["hit"] > hits else "ssd"
            if cache is None:
                cache, _ = prefill(model, r["hist"]); how = "recompute"
        cache, logits = prefill(model, r["turn"], cache)
        mx.eval(mx.argmax(logits, -1))
    ttfts.append(round(t.s, 3)); how_all.append(how)
    correct.append(greedy(model, cache, logits, 16) == r["ref"])
    del cache; mx.clear_cache()
stop.set()
log.update(resume_ttft_s=ttfts, resume_from=how_all, outputs_match=all(correct))
snap("after_resume")
log["app_retouch_after_resume_s"] = retouch()
snap("end")
hog.stdin.close(); hog.wait()
shutil.rmtree(tmp, ignore_errors=True)

K = ("swapouts_mb", "swapins_mb", "purged_mb")
d = lambda a, b: {k: round(log[b][k] - log[a][k]) for k in K}
log["delta_app_open"] = d("before_app", "after_retouch_idle")
log["delta_resume"] = d("after_retouch_idle", "end")
log["ssd_written_total_mb"] = round(ssd_written / 2**20)
log["pressure_events"] = events[:5] + events[-2:] if len(events) > 7 else events
os.makedirs("results/v2", exist_ok=True)
json.dump(log, open(f"results/v2/{mode}_{rep}.json", "w"), indent=1)
print(json.dumps({k: log[k] for k in ("mode", "rep", "app_retouch_idle_s", "app_retouch_after_resume_s",
      "delta_app_open", "ssd_written_at_idle_mb", "ssd_written_total_mb", "outputs_match")}
      | {"resume_total_s": round(sum(ttfts), 2), "from": {h: how_all.count(h) for h in set(how_all)},
         "footprint_idle_mb": log["before_app"]["footprint_mb"]}), flush=True)
