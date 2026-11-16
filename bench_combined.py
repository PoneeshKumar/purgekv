"""Exp F: all SSD-tier improvements together, on multi-turn chats.

Each chat: 4096-token history, then 8 turns of +256 tokens (real text), saved to SSD
after the history and after every turn. Policies (codec x write strategy):
  full+raw    rewrite the whole cache every save (what a plain save_prompt_cache tier does)
  full+lz     same, lossless split-plane LZFSE
  append+raw  write only new tokens' KV
  append+lz   append + lossless compression     (all lossless improvements combined)
  append+q8   append + 8-bit affine (lossy)
Then each chat resumes from SSD with the page cache bypassed (a true cold SSD read).
"""
import copy, json, os, shutil, sys, time
import mlx.core as mx
from mlx_lm import load
from purgekv.common import real_text_tokens, prefill, greedy, Timer
from purgekv.segstore import SegmentKVStore

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Llama-3.2-3B-Instruct-4bit"
CHATS, HIST, TURNS, TURN = 4, 4096, 8, 256
POLICIES = {"full+raw": ("raw", False), "full+lz": ("lz", False), "append+raw": ("raw", True),
            "append+lz": ("lz", True), "append+q8": ("q8", True)}
root = os.path.join(os.environ.get("SPILL_DIR", "/tmp"), "segbench")
shutil.rmtree(root, ignore_errors=True)
mx.set_cache_limit(256 << 20)
model, tok = load(MODEL)
stores = {p: SegmentKVStore(os.path.join(root, p), codec, app) for p, (codec, app) in POLICIES.items()}
save_s = {p: 0.0 for p in POLICIES}
rows = {p: {"resume_s": [], "match": [], "token_agreement": []} for p in POLICIES}
recompute_s = None

for chat in range(CHATS):
    cache, _ = prefill(model, real_text_tokens(tok, HIST, 10 + chat))
    for turn in range(TURNS + 1):
        if turn:
            cache, _ = prefill(model, real_text_tokens(tok, TURN, 1000 + chat * 50 + turn), cache)
        for p, st in stores.items():
            with Timer() as t:
                st.save(chat, cache)
            save_s[p] += t.s
    new = real_text_tokens(tok, 32, 5000 + chat)
    ref_c, logits = prefill(model, new, copy.deepcopy(cache))
    ref = greedy(model, ref_c, logits, 24)
    del ref_c
    if chat == 0:
        all_tokens = real_text_tokens(tok, HIST, 10)  # recompute baseline: full re-prefill
        hist_all = all_tokens + sum((real_text_tokens(tok, TURN, 1000 + t) for t in range(1, TURNS + 1)), [])
        with Timer() as t:
            c, lg = prefill(model, hist_all + new)
            mx.eval(mx.argmax(lg, -1))
        recompute_s = t.s
        del c
    del cache
    mx.clear_cache()
    for p, st in stores.items():
        with Timer() as t:
            c = st.load(chat)
            c, logits = prefill(model, new, c)
            mx.eval(mx.argmax(logits, -1))
        out = greedy(model, c, logits, 24)
        rows[p]["resume_s"].append(round(t.s, 3))
        rows[p]["match"].append(out == ref)
        rows[p]["token_agreement"].append(sum(a == b for a, b in zip(out, ref)) / len(ref))
        del c
        mx.clear_cache()
    print(f"chat {chat} done", flush=True)

tokens_final = HIST + TURNS * TURN
summary = {"model": MODEL, "chats": CHATS, "tokens_per_chat": tokens_final, "saves_per_chat": TURNS + 1,
           "recompute_ttft_s": round(recompute_s, 2), "policies": {}}
base = stores["full+raw"].bytes_written
for p, st in stores.items():
    on_disk = sum(os.path.getsize(os.path.join(st.root, f)) for f in os.listdir(st.root))
    r = rows[p]
    summary["policies"][p] = {
        "ssd_written_mb": round(st.bytes_written / 2**20), "vs_full_raw": round(base / st.bytes_written, 2),
        "on_disk_mb": round(on_disk / 2**20), "save_time_s": round(save_s[p], 2),
        "resume_cold_median_s": sorted(r["resume_s"])[len(r["resume_s"]) // 2],
        "identical_outputs": f"{sum(r['match'])}/{CHATS}",
        "token_agreement": round(sum(r["token_agreement"]) / CHATS, 3)}
    print(p, json.dumps(summary["policies"][p]), flush=True)
os.makedirs("results", exist_ok=True)
json.dump(summary, open(f"results/combined_{MODEL.split('/')[-1]}.json", "w"), indent=1)
shutil.rmtree(root, ignore_errors=True)
print("recompute_ttft_s", summary["recompute_ttft_s"])
