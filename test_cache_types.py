"""Round-trip every mlx-lm cache type through PurgeableKVStore and check that
generation after restore is token-identical to an uninterrupted run."""
import copy, json
import mlx.core as mx
from mlx_lm import load
from purgekv.common import real_text_tokens, prefill, greedy
from purgekv.store import PurgeableKVStore

CASES = [
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", False),   # KVCache
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", True),    # QuantizedKVCache (8-bit)
    ("mlx-community/gemma-3-1b-it-4bit", False),           # RotatingKVCache (sliding window) + KVCache
    ("mlx-community/Qwen3.5-0.8B-4bit", False),            # hybrid: ArraysCache (linear attn state) + KVCache
]
store, rows = PurgeableKVStore(), []
for model_id, quant in CASES:
    model, tok = load(model_id)
    hist = real_text_tokens(tok, 3000, session=0)
    turn = real_text_tokens(tok, 32, session=99)
    cache, _ = prefill(model, hist)
    if quant:
        cache = [c.to_quantized(group_size=64, bits=8) for c in cache]
    ref_cache = copy.deepcopy(cache)
    ref_cache, logits = prefill(model, turn, ref_cache)
    ref = greedy(model, ref_cache, logits, 24)

    nbytes = store.put("s", hist, cache)
    del cache
    restored, _ = store.get("s")
    restored, logits = prefill(model, turn, restored)
    out = greedy(model, restored, logits, 24)
    row = {"model": model_id.split("/")[-1] + (" +q8 KV" if quant else ""),
           "cache_types": sorted({type(c).__name__ for c in ref_cache}),
           "stored_mb": round(nbytes / 2**20, 1), "identical": out == ref}
    rows.append(row)
    print(json.dumps(row), flush=True)
    del model
    mx.clear_cache()
json.dump(rows, open("results/cache_types.json", "w"), indent=1)
