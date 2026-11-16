"""Exp H: layer-streamed resume vs load-then-compute (same I/O code), cold SSD reads.

  python bench_stream.py [MODEL]
Chats: 4096-token history + 8 appended 256-token turns (9 segments), saved with
SegmentKVStore (raw and q8). Resume with a new user turn of 32 / 256 / 1024 tokens.
TTFT = time from 'resume' to first token. Outputs checked against an uninterrupted run.
"""
import copy, json, os, shutil, sys, statistics as st
import mlx.core as mx
from mlx_lm import load
from purgekv.common import real_text_tokens, prefill, greedy, Timer
from purgekv.segstore import SegmentKVStore
from purgekv.stream import stream_prefill, load_then_compute

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Llama-3.2-3B-Instruct-4bit"
CHATS, HIST, TURNS, TURN, REPS = 2, 4096, 8, 256, 3
NEW = [32, 256, 1024]
root = os.path.join(os.environ.get("SPILL_DIR", "/tmp"), "streambench")
shutil.rmtree(root, ignore_errors=True)
mx.set_cache_limit(256 << 20)
model, tok = load(MODEL)
stores = {c: SegmentKVStore(os.path.join(root, c), c, True) for c in ("raw", "q8")}
refs = {}
for chat in range(CHATS):
    cache, _ = prefill(model, real_text_tokens(tok, HIST, 10 + chat))
    for turn in range(TURNS + 1):
        if turn:
            cache, _ = prefill(model, real_text_tokens(tok, TURN, 1000 + chat * 50 + turn), cache)
        for s in stores.values():
            s.save(chat, cache)
    for n in NEW:
        new = real_text_tokens(tok, n, 7000 + chat * 10 + n)
        c, lg = prefill(model, new, copy.deepcopy(cache))
        refs[(chat, n)] = (new, greedy(model, c, lg, 16))
        del c
    del cache; mx.clear_cache()

# warm up kernels for every shape
for n in NEW:
    for codec, s in stores.items():
        for fn in (stream_prefill, load_then_compute):
            c, lg = fn(model, s, 0, refs[(0, n)][0]); mx.eval(lg); del c
mx.clear_cache()

rows = []
for codec, s in stores.items():
    for n in NEW:
        res = {"codec": codec, "new_tokens": n}
        for name, fn in (("load_then_compute", load_then_compute), ("streamed", stream_prefill)):
            times, ok = [], True
            for rep in range(REPS):
                for chat in range(CHATS):
                    new, ref = refs[(chat, n)]
                    with Timer() as t:
                        c, lg = fn(model, s, chat, new)
                        mx.eval(mx.argmax(lg, -1))
                    times.append(t.s)
                    ok &= greedy(model, c, lg, 16) == ref if codec == "raw" else True
                    if codec == "q8":
                        res.setdefault(name + "_q8_agree", []).append(greedy(model, c, lg, 16) == ref)
                    del c; mx.clear_cache()
            res[name + "_ttft_s"] = round(st.median(times), 3)
            if codec == "raw":
                res[name + "_identical"] = ok
        res["speedup"] = round(res["load_then_compute_ttft_s"] / res["streamed_ttft_s"], 2)
        for k in [k for k in res if k.endswith("_q8_agree")]:
            res[k] = f"{sum(res[k])}/{len(res[k])}"
        rows.append(res)
        print(json.dumps(res), flush=True)

# compute-only reference: same new tokens with the cache already in RAM
for n in NEW:
    s = stores["raw"]
    c, _ = load_then_compute(model, s, 0, refs[(0, n)][0][:1])
    base = copy.deepcopy(c)
    ts = []
    for _ in range(REPS):
        c = copy.deepcopy(base)
        with Timer() as t:
            lg = model(mx.array(refs[(0, n)][0])[None], cache=c); mx.eval(lg)
        ts.append(t.s)
    print(json.dumps({"compute_only_s": round(st.median(ts), 3), "new_tokens": n}), flush=True)
    rows.append({"compute_only_s": round(st.median(ts), 3), "new_tokens": n})
json.dump(rows, open(f"results/stream_{MODEL.split('/')[-1]}.json", "w"), indent=1)
shutil.rmtree(root, ignore_errors=True)
