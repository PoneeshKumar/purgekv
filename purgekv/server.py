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
  * --kv-bits 8 (an mlx-lm flag) caches are stored as is: about half the bytes to write,
    store and read back. Outputs can differ slightly from full precision (mlx-lm's
    trade-off, not purgekv's).
"""
import argparse, json, logging, os, sys, time, uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from mlx_lm.models.cache import KVCache, LRUPromptCache, QuantizedKVCache

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

    def _disk_match(self, model, tokens, limit, partial=False):
        """Saved session sharing the longest start with `tokens` (at most `limit` tokens).

        partial=False: the session must be a whole prefix of `tokens` (to fork from it).
        partial=True: it may run longer and be trimmed back, like mlx-lm's RAM cache does.
        Chat templates often rewrite earlier turns (Qwen3 drops the empty <think></think>
        block from past replies), so the saved chat is rarely an exact prefix of the next
        request. Returns (session, number of tokens to reuse)."""
        self._store(model)
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
            key, n = self._disk_match(model, tokens, len(tokens) - 1, partial=True)  # leave >= 1 token
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
            extra = loaded[0].offset - n
            if extra > 0:  # the saved chat runs past the shared start: drop the tail
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
    def _save(self, model, tokens, cache):
        kinds = {type(c) for c in cache}
        if not cache or not (kinds <= {KVCache} or kinds <= {QuantizedKVCache}):
            return  # other cache layouts (sliding window, hybrid, draft models) stay RAM-only
        n = cache[0].offset
        if n <= 0 or len(tokens) < n or not all(isinstance(x, (int, np.integer)) for x in tokens[:n]):
            return
        toks = np.asarray(tokens[:n], np.int32)
        store = self._store(model)
        # The saved chat sharing the longest start with this one. Often it is not a whole
        # prefix: chat templates rewrite earlier turns (Qwen3 drops the empty <think></think>
        # from past replies), so the chats diverge inside the previous turn.
        src, m = self._disk_match(model, toks, n, partial=True)
        quant = isinstance(cache[0], QuantizedKVCache)
        if src is not None and any(("quant" in meta) != quant for _, meta in store.sessions[src]["segs"]):
            src, m = None, 0  # mlx-lm switched to quantized mid-chat: start a fresh file set
        src_n = len(self.tokens[(model, src)]) if src is not None else 0
        if src is not None and m == n == src_n:
            self.waiting.pop((model, src), None)
            self.waiting[(model, src)] = None
            return
        t = time.perf_counter()
        key = uuid.uuid4().hex[:16]
        shared = store.fork(src, key, upto=m) if src is not None else 0  # whole segments before the divergence
        # Write tokens shared..n, split at m: the next turn diverges after m (inside this
        # turn's reply), so it can keep the segment that ends at m.
        written = store.save(key, cache, upto=m) if shared < m else 0
        written += store.save(key, cache)
        tf = f"{key}.tok"
        toks.tofile(os.path.join(store.root, tf))
        store.sessions[key]["info"] = {"tokens": tf}
        self.tokens[(model, key)] = toks
        last = store.sessions[src]["segs"][-1][1]["shapes"][0][2] if src is not None else 0
        if src is not None and n >= src_n and m >= src_n - last:
            # The new chat is at least as long and repeats all of src but (at most) its last
            # saved segment, as rewritten by the template: src is superseded. Its shared files
            # live on in `key`. A chat that branches off earlier (an edited message) keeps src.
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
        store.delete(key)  # removes only files no other session uses
        if info.get("tokens"):
            try:
                os.remove(os.path.join(store.root, info["tokens"]))
            except FileNotFoundError:
                pass
        self.tokens.pop((model, key), None)
        self.waiting.pop((model, key), None)
        fut = self.prefetched.pop((model, key), None)
        if fut:
            fut[0].cancel()

    def _load(self, model, key):
        store = self._store(model)
        t = time.perf_counter()
        pf = self.prefetched.pop((model, key), None)
        if pf is not None:
            self.stats["prefetch_hits" if pf[0].done() else "prefetch_partial"] += 1
            raw = pf[0].result()
        else:
            raw = store.read_raw(key)
        cache = store.load(key, raw)
        self.stats["ssd_loads"] += 1
        self.stats["bytes_read"] += sum(b.size for _, _, b in raw)
        self.stats["load_s"] += time.perf_counter() - t
        log.info(f"purgekv: loaded {store.sessions[key]['saved']} tokens from SSD "
                 f"({'prefetched' if pf else 'cold'}) in {time.perf_counter() - t:.2f}s")
        return cache

    def _schedule(self):
        """Prefetch chats that are waiting for their next turn and are no longer in RAM.
        Runs on the generation thread; the I/O thread only reads files into numpy."""
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
                segs = list(store.sessions[k[1]]["segs"])
                nb = sum(os.path.getsize(p) for p, _ in segs)
                if used + nb > self.prefetch_bytes:
                    break
                self.prefetched[k] = (self._io.submit(store.read_raw, k[1], segs), nb)
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
