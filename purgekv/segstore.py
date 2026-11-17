"""SegmentKVStore: an SSD tier for standard KVCache that combines
  * append-only segments: each save writes only the tokens added since the last save
  * codec per segment: "raw", "lz" (lossless split-plane LZFSE) or "q8" (8-bit affine, lossy)
  * page-cache bypass (F_NOCACHE) on write and read so measured resumes are true SSD reads
"""
import fcntl, json, os
from concurrent.futures import ThreadPoolExecutor
import ctypes
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache

F_NOCACHE = 48
_lc = ctypes.CDLL("/usr/lib/libcompression.dylib")
for _f in (_lc.compression_encode_buffer, _lc.compression_decode_buffer):
    _f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    _f.restype = ctypes.c_size_t
LZFSE, CH = 0x801, 4 << 20
_pool = ThreadPoolExecutor(8)


def _enc(b):
    out = np.empty(b.size + 4096, np.uint8)
    n = _lc.compression_encode_buffer(out.ctypes.data, out.size, b.ctypes.data, b.size, None, LZFSE)
    return out[:n] if n else b


def _dec(blob, size):
    if blob.size == size:
        return blob
    out = np.empty(size, np.uint8)
    assert _lc.compression_decode_buffer(out.ctypes.data, size, blob.ctypes.data, blob.size, None, LZFSE) == size
    return out


def lz_pack(raw16: np.ndarray):
    """uint8 view of 16-bit values -> list of (blob, size) for hi plane then lo plane."""
    pairs = raw16.reshape(-1, 2)
    planes = [np.ascontiguousarray(pairs[:, 1]), np.ascontiguousarray(pairs[:, 0])]
    jobs = [p[i:i + CH] for p in planes for i in range(0, p.size, CH)]
    return list(zip(_pool.map(_enc, jobs), [j.size for j in jobs]))


def lz_unpack(parts, nbytes):
    outs = list(_pool.map(lambda bs: _dec(*bs), parts))
    half = len(outs) // 2
    hi, lo = np.concatenate(outs[:half]), np.concatenate(outs[half:])
    return np.stack([lo, hi], 1).reshape(-1)[:nbytes]


def _write(path, chunks):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    n = 0
    try:
        for c in chunks:
            n += os.write(fd, memoryview(np.ascontiguousarray(c)).cast("B"))
    finally:
        os.close(fd)
    return n


def _read(path):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    try:
        size = os.fstat(fd).st_size
        buf = np.empty(size, np.uint8)
        mv, got = memoryview(buf), 0
        while got < size:
            got += os.readv(fd, [mv[got:]])
    finally:
        os.close(fd)
    return buf


INDEX = "index.json"


def _meta_to_json(meta):
    m = dict(meta)
    m["dtype"] = str(meta["dtype"]).split(".")[-1]
    if "qdtypes" in m:
        m["qdtypes"] = [[str(d).split(".")[-1] for d in t] for t in meta["qdtypes"]]
    return m


def _meta_from_json(m):
    m = dict(m)
    m["dtype"] = getattr(mx, m["dtype"])
    m["shapes"] = [tuple(s) for s in m["shapes"]]
    if "qdtypes" in m:
        m["qdtypes"] = [[getattr(mx, d) for d in t] for t in m["qdtypes"]]
        m["qshapes"] = [[tuple(s) for s in t] for t in m["qshapes"]]
    return m


class SegmentKVStore:
    def __init__(self, root, codec="raw", append=True):
        self.root, self.codec, self.append = root, codec, append
        os.makedirs(root, exist_ok=True)
        self.sessions = {}  # key -> {"saved": n_tokens, "segs": [(path, meta)]}
        self.bytes_written = 0

    # ---- shared cached prompts -------------------------------------------------
    def fork(self, src, dst):
        """Start session `dst` from cached session `src` (e.g. a long system prompt or
        document) without copying it: dst references src's segment files and later
        saves to dst write only dst's own new tokens."""
        s = self.sessions[src]
        self.sessions[dst] = {"saved": s["saved"], "segs": list(s["segs"])}

    def _referenced(self, path, ignore=None):
        return any(path == p for k, s in self.sessions.items() if k != ignore for p, _ in s["segs"])

    def delete(self, key):
        """Forget a session; remove only the segment files no other session uses."""
        s = self.sessions.pop(key, None)
        for p, _ in (s or {}).get("segs", []):
            if not self._referenced(p) and os.path.exists(p):
                os.remove(p)

    def compact(self, key):
        """Merge a session's own segments into one file (call while the chat is idle).

        Saving after every turn leaves many small segments, and resume reads each one
        per layer: 64 segments loaded ~6x slower than 1 on an M4. Segments shared with
        other sessions (a forked prompt) are left alone so they stay shared.
        Returns the number of segments merged."""
        if self.codec != "raw":
            raise NotImplementedError("compact() supports codec='raw'")
        segs = self.sessions[key]["segs"]
        first = 0
        while first < len(segs) and self._referenced(segs[first][0], ignore=key):
            first += 1  # leading shared prompt segments stay as they are
        own = segs[first:]
        if len(own) < 2:
            return 0
        n_tensors = len(own[0][1]["shapes"])
        start = sum(m["shapes"][0][2] for _, m in segs[:first])
        n = start + sum(m["shapes"][0][2] for _, m in own)
        path = os.path.join(self.root, f"{key}_{start}_{n}.c.{self.codec}")
        out = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        fcntl.fcntl(out, F_NOCACHE, 1)
        shapes, written = [], 0
        try:
            for t in range(n_tensors):  # one tensor in memory at a time
                B, Hh, _, Dd = own[0][1]["shapes"][t]
                T = sum(m["shapes"][t][2] for _, m in own)
                buf = np.empty((B, Hh, T, Dd), np.uint16)
                t0 = 0
                for p, meta in own:
                    off = sum(int(np.prod(s)) * 2 for s in meta["shapes"][:t])
                    ln = meta["shapes"][t][2]
                    fd = os.open(p, os.O_RDONLY)
                    try:
                        fcntl.fcntl(fd, F_NOCACHE, 1)
                        views = [memoryview(buf[0, h, t0:t0 + ln, :]).cast("B") for h in range(Hh)]
                        if os.preadv(fd, views, off) != sum(v.nbytes for v in views):
                            raise OSError(f"short read compacting {p}")
                    finally:
                        os.close(fd)
                    t0 += ln
                mv = memoryview(buf.reshape(-1).view(np.uint8))
                while mv:
                    k = os.write(out, mv)
                    written += k
                    mv = mv[k:]
                shapes.append(buf.shape)
        finally:
            os.close(out)
        os.replace(path + ".tmp", path)  # never leave a half-written segment behind
        meta = {"shapes": shapes, "dtype": own[0][1]["dtype"], "nbytes": written}
        self.sessions[key]["segs"] = segs[:first] + [(path, meta)]
        for p, _ in own:
            if not self._referenced(p) and os.path.exists(p):
                os.remove(p)
        self.bytes_written += written
        return len(own)

    # ---- survive a restart -----------------------------------------------------
    def save_index(self):
        """Write the session index to <root>/index.json (atomically)."""
        data = {"codec": self.codec, "append": self.append, "sessions": {
            k: {"saved": s["saved"],
                "segs": [(os.path.relpath(p, self.root), _meta_to_json(m)) for p, m in s["segs"]]}
            for k, s in self.sessions.items()}}
        tmp = os.path.join(self.root, INDEX + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(self.root, INDEX))

    @classmethod
    def open(cls, root):
        """Reopen a store written by save_index(), e.g. after the process restarted."""
        with open(os.path.join(root, INDEX)) as f:
            data = json.load(f)
        store = cls(root, data["codec"], data["append"])
        store.sessions = {k: {"saved": s["saved"],
                              "segs": [(os.path.join(root, p), _meta_from_json(m)) for p, m in s["segs"]]}
                          for k, s in data["sessions"].items()}
        return store

    def save(self, key, cache):
        """Persist tokens not yet saved (or everything, if append=False)."""
        s = self.sessions.setdefault(key, {"saved": 0, "segs": []})
        n = cache[0].offset
        start = s["saved"] if self.append else 0
        if not self.append:
            old, s["segs"] = s["segs"], []
            for p, _ in old:  # a full rewrite must never delete a shared prompt's files
                if not self._referenced(p) and os.path.exists(p):
                    os.remove(p)
        if n <= start:
            return 0
        layers = []
        for c in cache:
            layers += [c.keys[..., start:n, :], c.values[..., start:n, :]]
        mx.eval(layers)
        path = os.path.join(self.root, f"{key}_{start}_{n}.{self.codec}")
        meta = {"shapes": [a.shape for a in layers], "dtype": layers[0].dtype}
        if self.codec == "q8":
            q = [mx.quantize(a, group_size=64, bits=8) for a in layers]
            mx.eval(q)
            # numpy has no bfloat16: move bf16 scales/biases through a uint16 view
            arrs = [np.asarray(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x).reshape(-1).view(np.uint8)
                    for t in q for x in t]
            meta["sizes"] = [a.size for a in arrs]
            meta["qshapes"] = [[x.shape for x in t] for t in q]
            meta["qdtypes"] = [[x.dtype for x in t] for t in q]
            written = _write(path, arrs)
        else:
            raw = np.concatenate([np.asarray(a.view(mx.uint16)).reshape(-1).view(np.uint8) for a in layers])
            meta["nbytes"] = raw.size
            if self.codec == "lz":
                parts = lz_pack(raw)
                meta["parts"] = [(b.size, sz) for b, sz in parts]
                written = _write(path, [b for b, _ in parts])
            else:
                written = _write(path, [raw])
        s["segs"].append((path, meta))
        s["saved"] = n
        self.bytes_written += written
        return written

    def load(self, key):
        segs = self.sessions[key]["segs"]
        per_layer = None
        for path, meta in segs:
            buf = _read(path)
            if self.codec == "q8":
                arrs, off = [], 0
                for sz in meta["sizes"]:
                    arrs.append(buf[off:off + sz]); off += sz
                tensors, i = [], 0
                for shapes, dts in zip(meta["qshapes"], meta["qdtypes"]):
                    w, sc, bi = (mx.array(arrs[i + j]).view(dts[j]).reshape(shapes[j]) for j in range(3))
                    tensors.append(mx.dequantize(w, sc, bi, group_size=64, bits=8).astype(meta["dtype"]))
                    i += 3
            else:
                if self.codec == "lz":
                    parts, off = [], 0
                    for bsz, sz in meta["parts"]:
                        parts.append((buf[off:off + bsz], sz)); off += bsz
                    raw = lz_unpack(parts, meta["nbytes"])
                else:
                    raw = buf
                whole = mx.array(raw)
                tensors, off = [], 0
                for shp in meta["shapes"]:
                    nb = int(np.prod(shp)) * 2
                    tensors.append(whole[off:off + nb].view(meta["dtype"]).reshape(shp)); off += nb
            per_layer = [[t] for t in tensors] if per_layer is None else [p + [t] for p, t in zip(per_layer, tensors)]
        per_layer = [p[0] if len(p) == 1 else mx.concatenate(p, axis=2) for p in per_layer]
        cache = []
        for i in range(0, len(per_layer), 2):
            c = KVCache()
            c.keys, c.values, c.offset = per_layer[i], per_layer[i + 1], per_layer[i].shape[2]
            cache.append(c)
        mx.eval([x for c in cache for x in (c.keys, c.values)])
        return cache
