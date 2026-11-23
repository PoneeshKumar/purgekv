"""purgekv-server: mlx_lm.server with an SSD tier for its prompt cache.

  purgekv-server --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
                 --prompt-cache-bytes 4GB [--kv-dir DIR] [--prefetch-gb 2] [--kv-bits 8]

Every flag of `mlx_lm.server` works; the OpenAI-compatible API is unchanged, so any
client (chat apps, coding agents) gets this with no code changes. What it adds:

  * SSD tier: after every reply the chat's KV cache is appended to disk (only the tokens
    that are new since the last save; chats that share a prefix share its files).
    Chats pushed out of RAM by --prompt-cache-bytes, or lost to a server restart, are
    reloaded from SSD instead of recomputed.
  * Think-time prefetch: a chat that just got a reply is waiting on its user or on a
    tool (agents). If it is pushed out of RAM, the server reads it back from SSD in the
    background, oldest-waiting first, up to --prefetch-gb, so the next request does
    not wait for the SSD.
  * Any model mlx_lm.server runs: full-attention layers are appended a turn at a time;
    sliding-window and recurrent layers (Gemma 3, Qwen3.5) are saved whole per turn.
  * --kv-bits 8 (an mlx-lm flag) caches are stored as is: about half the bytes to write,
    store and read back. Outputs can differ slightly from full precision (mlx-lm's
    trade-off, not purgekv's).
"""
import argparse, json, logging, os, sys, time, uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from mlx_lm.models.cache import (KVCache, LRUPromptCache, QuantizedKVCache, can_trim_prompt_cache,
                                  load_prompt_cache, save_prompt_cache)

from .segstore import SegmentKVStore, INDEX

CONFIG = {"kv_dir": os.path.expanduser("~/.cache/purgekv/server"), "prefetch_bytes": 2 << 30,
          "ram_bytes": None}
log = logging.getLogger("purgekv")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))[:120]


class SSDPromptCache(LRUPromptCache):
    """Drop-in for mlx_lm.models.cache.LRUPromptCache: same RAM LRU, plus an SSD tier
    under it and background prefetch of chats that are waiting for their next turn."""

    def __init__(self, max_size=10, max_bytes=1 << 63, kv_dir=None, prefetch_bytes=None):
        super().__init__(max_size, CONFIG["ram_bytes"] or max_bytes)
        self.root = kv_dir or CONFIG["kv_dir"]
        self.prefetch_bytes = CONFIG["prefetch_bytes"] if prefetch_bytes is None else prefetch_bytes
        self.stores = {}      # model key -> SegmentKVStore
        self.tokens = {}      # (model key, session) -> np.int32 tokens covered by the saved cache
        self.waiting = OrderedDict()   # (model key, session) -> None, oldest reply first
        self.prefetched = {}  # (model key, session) -> (Future, nbytes)
        self._io = ThreadPoolExecutor(1, thread_name_prefix="purgekv-prefetch")
        self.stats = {"ram_hits": 0, "ssd_loads": 0, "prefetch_hits": 0, "prefetch_partial": 0,
                      "saves": 0, "bytes_written": 0, "bytes_read": 0, "save_s": 0.0, "load_s": 0.0}

    # ---- per-model stores, reopened after a restart ---------------------------
    def _store(self, model):
        if model not in self.stores:
            root = os.path.join(self.root, _safe(model))
            if os.path.exists(os.path.join(root, INDEX)):
                st = SegmentKVStore.open(root)
                for k, s in st.sessions.items():
                    tf = os.path.join(root, s["info"].get("tokens", ""))
                    if os.path.isfile(tf):
                        self.tokens[(model, k)] = np.fromfile(tf, np.int32)
                st.sessions = {k: s for k, s in st.sessions.items() if (model, k) in self.tokens}
                log.info(f"purgekv: reopened {len(st.sessions)} cached chats from {root}")
            else:
                st = SegmentKVStore(root)
            self.stores[model] = st
        return self.stores[model]

    def _disk_match(self, model, tokens, limit, partial=False, load=False):
        """Saved session sharing the longest start with `tokens` (at most `limit` tokens).

        partial=False: the session must be a whole prefix of `tokens` (to fork from it).
        partial=True: it may run longer and be trimmed back, like mlx-lm's RAM cache does.
        Chat templates often rewrite earlier turns (Qwen3 drops the empty <think></think>
        block from past replies), so the saved chat is rarely an exact prefix of the next
        request. load=True: a session whose cache cannot be trimmed only counts if all of it
        is reused. Returns (session, number of tokens to reuse)."""
        store = self._store(model)
        best, best_n = None, 0
        if limit <= 0:
            return None, 0
        t = np.asarray(tokens[:limit], np.int64)
        for (m, k), tok in self.tokens.items():
            if m != model:
                continue
            n = min(len(tok), len(t))
            if not partial and len(tok) > len(t):
                continue
            eq = t[:n] == tok[:n]
            common = n if eq.all() else int(np.argmin(eq))
            if not partial and common < len(tok):
                continue
            if load and common < len(tok) and not store.sessions[k].get("info", {}).get("trim", True):
                continue
            if common > best_n:
                best, best_n = k, common
        return best, best_n

    def _in_ram(self, model, tok):
        r = self._trie.search(model, tok.tolist())
        return r.exact is not None or (r.longer is not None and r.common_prefix >= len(tok))

    # ---- LRUPromptCache interface ---------------------------------------------
    def fetch_nearest_cache(self, model, tokens):
        cache, rest = super().fetch_nearest_cache(model, tokens)
        ram_n = len(tokens) - len(rest)
        try:
            key, n = self._disk_match(model, tokens, len(tokens) - 1, partial=True, load=True)  # leave >= 1 token
        except Exception:
            log.exception("purgekv: SSD lookup failed")
            key, n = None, 0
        if key is not None:
            self.waiting.pop((model, key), None)
        if key is None or n <= ram_n:
            if cache is not None:
                self.stats["ram_hits"] += 1
            self._schedule()
            return cache, rest
        try:
            loaded = self._load(model, key)
            extra = self._length(loaded) - n
            if extra > 0:  # the saved chat runs past the shared start: drop the tail
                if not can_trim_prompt_cache(loaded):
                    raise RuntimeError("cannot trim this cache")  # _disk_match should prevent this
                for c in loaded:
                    c.trim(extra)
        except Exception:
            log.exception("purgekv: SSD load failed; recomputing instead")
            self._schedule()
            return cache, rest
        self._schedule()
        self._dump()
        return loaded, tokens[n:]

    def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
        try:
            self._save(model, tokens, prompt_cache)
        except Exception:
            log.exception("purgekv: SSD save failed (the RAM cache is unaffected)")
        self._schedule()
        self._dump()

    def _dump(self):
        """Counters in <kv-dir>/stats.json, for benchmarks and curiosity."""
        try:
            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, "stats.json.tmp"), "w") as f:
                json.dump(self.stats, f)
            os.replace(os.path.join(self.root, "stats.json.tmp"), os.path.join(self.root, "stats.json"))
        except OSError:
            pass

    def trim_to(self, *, n_sequences=None, n_bytes=None):
        super().trim_to(n_sequences=n_sequences, n_bytes=n_bytes)
        self._schedule()

    # ---- SSD tier -------------------------------------------------------------
    # A chat's cache is a list of per-layer caches. Layers that grow with the chat
    # (KVCache, QuantizedKVCache) go to the segment store: only new tokens are written,
    # and chats with a common start share its files. Every other layer type (sliding
    # window, linear attention / recurrent state, anything mlx-lm adds later) is saved
    # whole with mlx-lm's own save_prompt_cache, one snapshot per saved chat. Those
    # layers have a fixed size, so the snapshot stays small however long the chat is.
    # A chat with layers that cannot be trimmed is only reused at exactly the length it
    # was saved, as in mlx-lm's RAM cache, which also checkpoints at the end of each prompt.

    @staticmethod
    def _length(cache):
        for c in cache:
            if isinstance(getattr(c, "offset", None), (int, np.integer)):
                return int(c.offset)
        return None  # e.g. a pure recurrent model: no token count to key the save on

    def _save(self, model, tokens, cache):
        n = self._length(cache) if cache else None
        if not n or len(tokens) < n or not all(isinstance(x, (int, np.integer)) for x in tokens[:n]):
            return
        layout = ["kv" if type(c) in (KVCache, QuantizedKVCache) else "snap" for c in cache]
        kv = [c for c, k in zip(cache, layout) if k == "kv"]
        snap = [c for c, k in zip(cache, layout) if k == "snap"]
        if len({type(c) for c in kv}) > 1:
            return  # mixed full-precision and quantized layers: leave it to RAM
        toks = np.asarray(tokens[:n], np.int32)
        store = self._store(model)
        trimmable = can_trim_prompt_cache(cache)
        # The saved chat sharing the longest start with this one. Often it is not a whole
        # prefix: chat templates rewrite earlier turns (Qwen3 drops the empty <think></think>
        # from past replies), so the chats diverge inside the previous turn.
        src, m = self._disk_match(model, toks, n, partial=True)
        quant = bool(kv) and isinstance(kv[0], QuantizedKVCache)
        if src is not None and (store.sessions[src]["info"].get("layout", layout) != layout or
                                any(("quant" in meta) != quant for _, meta in store.sessions[src]["segs"])):
            src, m = None, 0  # a different cache layout (mlx-lm switched to quantized): start fresh
        src_n = len(self.tokens[(model, src)]) if src is not None else 0
        if src is not None and m == n == src_n:
            self.waiting.pop((model, src), None)
            self.waiting[(model, src)] = None
            return
        t = time.perf_counter()
        key = uuid.uuid4().hex[:16]
        written, info = 0, {"tokens": f"{key}.tok", "layout": layout, "trim": trimmable}
        if snap:  # write it first: if a layer cannot be serialized, nothing is half-saved
            info["snap"] = f"{key}.snap.safetensors"
            path = os.path.join(store.root, info["snap"])
            save_prompt_cache(path, snap)
            written += os.path.getsize(path)
        shared = store.fork(src, key, upto=m) if src is not None else 0  # whole segments before the divergence
        if kv:
            # Write tokens shared..n, split at m: the next turn diverges after m (inside this
            # turn's reply), so it can keep the segment that ends at m.
            written += store.save(key, kv, upto=m) if shared < m else 0
            written += store.save(key, kv)
        else:
            store.sessions[key] = {"saved": n, "segs": []}
        toks.tofile(os.path.join(store.root, info["tokens"]))
        store.sessions[key]["info"] = info
        self.tokens[(model, key)] = toks
        segs = store.sessions[src]["segs"] if src is not None else []
        last = segs[-1][1]["shapes"][0][2] if segs else src_n
        if src is not None and trimmable and n >= src_n and m >= src_n - last:
            # The new chat is at least as long and repeats all of src but (at most) its last
            # saved segment, as rewritten by the template: src is superseded. Its shared files
            # live on in `key`. A chat that branches off earlier (an edited message) keeps src.
            # Untrimmable chats keep it: only an exact-length checkpoint can be reused.
            self._forget(model, src)
        store.save_index()
        self.waiting[(model, key)] = None
        while len(self.waiting) > 256:
            self.waiting.popitem(last=False)
        self.stats["saves"] += 1
        self.stats["bytes_written"] += written
        self.stats["save_s"] += time.perf_counter() - t

    def _forget(self, model, key):
        store = self.stores[model]
        info = store.sessions.get(key, {}).get("info", {})
        store.delete(key)  # removes only segment files no other session uses
        for f in (info.get("tokens"), info.get("snap")):
            if f:
                try:
                    os.remove(os.path.join(store.root, f))
                except FileNotFoundError:
                    pass
        self.tokens.pop((model, key), None)
        self.waiting.pop((model, key), None)
        fut = self.prefetched.pop((model, key), None)
        if fut:
            fut[0].cancel()

    @staticmethod
    def _read(store, key, segs, snap):
        """File I/O only (no MLX): safe on the prefetch thread."""
        raw = store.read_raw(key, segs)
        if snap:  # read it once so the load below comes from the page cache
            with open(snap, "rb") as f:
                while f.read(1 << 24):
                    pass
        return raw

    def _files(self, model, key):
        store = self.stores[model]
        s = store.sessions[key]
        snap = s["info"].get("snap")
        return list(s["segs"]), os.path.join(store.root, snap) if snap else None

    def _load(self, model, key):
        store = self._store(model)
        t = time.perf_counter()
        segs, snap = self._files(model, key)
        pf = self.prefetched.pop((model, key), None)
        if pf is not None:
            self.stats["prefetch_hits" if pf[0].done() else "prefetch_partial"] += 1
            raw = pf[0].result()
        else:
            raw = self._read(store, key, segs, snap)
        layout = store.sessions[key]["info"].get("layout")
        kv = store.load(key, raw) if segs else []
        if layout is None:  # saved by 0.3: every layer is a KVCache
            cache = kv
        else:
            snaps = iter(load_prompt_cache(snap) if snap else [])
            kvs = iter(kv)
            cache = [next(kvs) if k == "kv" else next(snaps) for k in layout]
        self.stats["ssd_loads"] += 1
        self.stats["bytes_read"] += sum(b.size for _, _, b in raw) + (os.path.getsize(snap) if snap else 0)
        self.stats["load_s"] += time.perf_counter() - t
        log.info(f"purgekv: loaded {store.sessions[key]['saved']} tokens from SSD "
                 f"({'prefetched' if pf else 'cold'}) in {time.perf_counter() - t:.2f}s")
        return cache

    def _schedule(self):
        """Prefetch chats that are waiting for their next turn and are no longer in RAM.
        Runs on the generation thread; the I/O thread only reads files."""
        if self.prefetch_bytes <= 0:
            return
        try:
            for k in [k for k in self.prefetched if k not in self.waiting or self._in_ram(k[0], self.tokens[k])]:
                self.prefetched.pop(k)[0].cancel()  # consumed elsewhere or back in RAM
            used = sum(nb for _, nb in self.prefetched.values())
            for k in self.waiting:  # oldest reply first: in an agent loop it comes back first
                if k in self.prefetched or self._in_ram(k[0], self.tokens[k]):
                    continue
                store = self.stores[k[0]]
                segs, snap = self._files(*k)
                nb = sum(os.path.getsize(p) for p, _ in segs) + (os.path.getsize(snap) if snap else 0)
                if used + nb > self.prefetch_bytes:
                    break
                self.prefetched[k] = (self._io.submit(self._read, store, k[1], segs, snap), nb)
                used += nb
        except Exception:
            log.exception("purgekv: prefetch scheduling failed")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--kv-dir", default=CONFIG["kv_dir"])
    ap.add_argument("--prefetch-gb", type=float, default=2.0)
    ap.add_argument("--prompt-cache-bytes", default=None)
    ours, rest = ap.parse_known_args(argv)
    if "-h" in rest or "--help" in rest:
        print(__doc__)
    CONFIG["kv_dir"] = os.path.expanduser(ours.kv_dir)
    CONFIG["prefetch_bytes"] = int(ours.prefetch_gb * (1 << 30))
    if ours.prompt_cache_bytes is not None:
        from mlx_lm.server import _parse_size
        CONFIG["ram_bytes"] = _parse_size(ours.prompt_cache_bytes)
        rest += ["--prompt-cache-bytes", ours.prompt_cache_bytes]
    import mlx_lm.server as srv
    srv.LRUPromptCache = SSDPromptCache  # run() builds the server's prompt cache from this name
    sys.argv = [sys.argv[0]] + rest
    srv.main()


if __name__ == "__main__":
    main()
