"""PurgeableKVStore: a prompt/KV cache tier the macOS kernel can reclaim for free.

Idle sessions' caches are packed into one purgeable VM object each and marked
VOLATILE. While memory is plentiful a resume costs one memcpy. Under pressure
the kernel discards the object instead of swapping it (and instead of swapping
other apps); the next resume sees EMPTY and the caller falls back.

Works with any mlx-lm cache class (KVCache, QuantizedKVCache, RotatingKVCache,
ArraysCache, CacheList...) by packing the arrays in `cache.state` and keeping
the non-array leaves (offsets, etc.) in Python.
"""
import os, threading, functools
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .purgeable import PurgeableRegion, VOLATILE

# Group 7 is purged last. Group 0 (the default) can be purged almost
# immediately on a Mac whose free-page pool is low, the normal state of macOS.
VOLATILE_FLAGS = VOLATILE | (7 << 8)
ALIGN = 64


def _raw(a: mx.array) -> np.ndarray:
    a = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
    return np.asarray(a).reshape(-1).view(np.uint8)


def _locked(f):
    @functools.wraps(f)
    def g(self, *a, **k):
        with self.lock:
            return f(self, *a, **k)
    return g


class PurgeableKVStore:
    def __init__(self, volatile_flags: int = VOLATILE_FLAGS):
        self.flags = volatile_flags
        self.entries = {}  # key -> (region, classes, leaves, tokens)
        self.lock = threading.RLock()
        self.spills = {}   # key -> SSD copy path
        self.stats = {"hit": 0, "spill": 0, "purged": 0, "miss": 0}

    @_locked
    def put(self, key, tokens, prompt_cache):
        self.drop(key)
        flat = tree_flatten([c.state for c in prompt_cache])
        arrays = [v for _, v in flat if isinstance(v, mx.array)]
        mx.eval(arrays)
        leaves, total = [], 0
        for path, v in flat:
            if isinstance(v, mx.array):
                leaves.append((path, ("arr", total, v.shape, v.dtype, v.nbytes)))
                total += -(-v.nbytes // ALIGN) * ALIGN
            else:
                leaves.append((path, ("py", v)))
        region = PurgeableRegion(total)
        buf = np.frombuffer(region.buffer(), dtype=np.uint8)
        for (_, leaf), v in zip(leaves, (v for _, v in flat)):
            if leaf[0] == "arr":
                buf[leaf[1]:leaf[1] + leaf[4]] = _raw(v)
        region._set(self.flags)
        self.entries[key] = (region, [type(c) for c in prompt_cache], leaves, list(tokens))
        return total

    @_locked
    def get(self, key):
        """Returns (prompt_cache, tokens); prompt_cache is None if the kernel purged it."""
        if key not in self.entries:
            self.stats["miss"] += 1
            return None, None
        region, classes, leaves, tokens = self.entries.pop(key)
        spill = self.spills.pop(key, None)
        if region.pin():
            whole = mx.array(np.frombuffer(region.buffer(), dtype=np.uint8))  # one bulk copy
            self.stats["hit"] += 1
        elif spill:
            whole = mx.array(np.fromfile(spill, dtype=np.uint8))
            self.stats["spill"] += 1
        else:
            region.free()
            self.stats["purged"] += 1
            return None, tokens
        mx.eval(whole)
        region.free()
        if spill:
            os.remove(spill)
        flat = []
        for path, leaf in leaves:
            if leaf[0] == "arr":
                _, start, shape, dt, nb = leaf
                flat.append((path, whole[start:start + nb].view(dt).reshape(shape)))
            else:
                flat.append((path, leaf[1]))
        states = tree_unflatten(flat)
        return [cls.from_state(s) for cls, s in zip(classes, states)], tokens

    @_locked
    def spill(self, key, path):
        """Write a still-resident entry to SSD so a later purge is cheap.
        Returns bytes written (0 if already purged or already spilled)."""
        if key not in self.entries or key in self.spills:
            return 0
        region = self.entries[key][0]
        if not region.pin():
            return 0
        try:
            np.frombuffer(region.buffer(), dtype=np.uint8).tofile(path)
        finally:
            region._set(self.flags)
        self.spills[key] = path
        return region.size

    def resident(self, key):
        """Peek whether an entry is still resident, without pinning it."""
        return key in self.entries and (self.entries[key][0].state() & 3) != 2

    @_locked
    def drop(self, key):
        e = self.entries.pop(key, None)
        if e:
            e[0].free()
        p = self.spills.pop(key, None)
        if p and os.path.exists(p):
            os.remove(p)
