"""Exp E: can lossless compression shrink what the SSD tier writes and reads?

Real KV caches from real text, compressed with Apple's built-in libcompression
(LZ4 / LZFSE / ZLIB), either on raw bytes or after splitting each 16-bit value into
a high-byte plane (sign + exponent + top mantissa bits, low entropy) and a low-byte
plane. Throughput uses 8 threads over 4 MB chunks (ctypes releases the GIL).
Every result is round-tripped and checked bit-for-bit.
"""
import ctypes, json, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from purgekv.common import real_text_tokens, prefill

lc = ctypes.CDLL("/usr/lib/libcompression.dylib")
for f in (lc.compression_encode_buffer, lc.compression_decode_buffer):
    f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    f.restype = ctypes.c_size_t
ALGS = {"lz4": 0x100, "lzfse": 0x801, "zlib": 0x205}
CH, POOL = 4 << 20, ThreadPoolExecutor(8)


def enc(buf: np.ndarray, alg):
    out = np.empty(buf.size + 4096, np.uint8)
    n = lc.compression_encode_buffer(out.ctypes.data, out.size, buf.ctypes.data, buf.size, None, alg)
    return out[:n] if n else buf.copy()  # 0 = didn't fit: store raw


def dec(blob: np.ndarray, size, alg):
    if blob.size == size:
        return blob
    out = np.empty(size, np.uint8)
    n = lc.compression_decode_buffer(out.ctypes.data, size, blob.ctypes.data, blob.size, None, alg)
    assert n == size
    return out


def run(raw: np.ndarray, split: bool, alg):
    """raw: uint8 view of 16-bit data. Returns (ratio, enc GB/s, dec GB/s, lossless)."""
    planes = [raw.reshape(-1, 2)[:, 1].copy(), raw.reshape(-1, 2)[:, 0].copy()] if split else [raw]
    chunks = [(p, i) for p in planes for i in range(0, p.size, CH)]
    t = time.perf_counter()
    blobs = list(POOL.map(lambda c: enc(c[0][c[1]:c[1] + CH], alg), chunks))
    te = time.perf_counter() - t
    t = time.perf_counter()
    outs = list(POOL.map(lambda bc: dec(bc[0], min(CH, bc[1][0].size - bc[1][1]), alg), zip(blobs, chunks)))
    td = time.perf_counter() - t
    rebuilt = [np.concatenate(outs[:len(outs) // len(planes)]), np.concatenate(outs[len(outs) // len(planes):])] if split else [np.concatenate(outs)]
    back = np.stack([rebuilt[1], rebuilt[0]], 1).reshape(-1) if split else rebuilt[0]
    gb = raw.size / 1e9
    return raw.size / sum(b.size for b in blobs), gb / te, gb / td, bool(np.array_equal(back, raw))


def entropy_bits(a: np.ndarray):
    p = np.bincount(a, minlength=256) / a.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


rows = []
for model_id in sys.argv[1:]:
    model, tok = load(model_id)
    cache, _ = prefill(model, real_text_tokens(tok, 8192, 3))
    arrs = [v for _, v in tree_flatten([c.state for c in cache]) if isinstance(v, mx.array)]
    dtype = str(arrs[0].dtype)
    raw = np.concatenate([np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a.view(mx.uint16))
                          .reshape(-1).view(np.uint8) for a in arrs])
    hi = raw.reshape(-1, 2)[:, 1]
    base = {"model": model_id.split("/")[-1], "dtype": dtype, "mb": round(raw.size / 2**20),
            "high_byte_entropy_bits": round(entropy_bits(hi), 2),
            "ideal_ratio_hi_only": round(16 / (8 + entropy_bits(hi)), 3)}
    for split in (False, True):
        for name, alg in ALGS.items():
            r, e, d, ok = run(raw, split, alg)
            row = base | {"method": ("split+" if split else "") + name, "ratio": round(r, 3),
                          "enc_GBps": round(e, 2), "dec_GBps": round(d, 2), "lossless": ok}
            rows.append(row); print(json.dumps(row), flush=True)
    del model, cache; mx.clear_cache()
json.dump(rows, open("results/compress.json", "w"), indent=1)
