"""Layer-streamed resume: overlap SSD reads of layer i+1's KV with the GPU computing layer i.

Today's Mac SSD tiers load the whole cache, then run the model, so
TTFT = load + compute. Streaming makes it ~max(load, compute) + one layer.
On Apple Silicon the bytes read from SSD land in memory the GPU uses directly
(unified memory), so there is no host-to-device copy stage to pipeline as on CUDA.

Works with SegmentKVStore files (raw or q8 codec) and decoder-only mlx-lm models with
standard attention: tested on Llama 3.x and Qwen3 (check_supported() refuses others).
"""
import fcntl, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

F_NOCACHE = 48
_pool = ThreadPoolExecutor(4)


def _layout(store, key):
    """Per segment: path and, per tensor (k0, v0, k1, v1, ...), the byte ranges to read."""
    out = []
    for path, meta in store.sessions[key]["segs"]:
        ranges, off = [], 0
        if store.codec == "q8":
            sizes = iter(meta["sizes"])
            for shapes, dts in zip(meta["qshapes"], meta["qdtypes"]):
                parts = []
                for shp, dt in zip(shapes, dts):
                    sz = next(sizes); parts.append((off, sz, shp, dt)); off += sz
                ranges.append(parts)
        else:
            for shp in meta["shapes"]:
                nb = int(np.prod(shp)) * 2
                ranges.append([(off, nb, shp, meta["dtype"])]); off += nb
        out.append((path, ranges, meta["dtype"]))
    return out


def _pread_all(path, spans):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    try:
        bufs = []
        for off, sz, _, _ in spans:
            b = np.empty(sz, np.uint8)
            mv, got = memoryview(b), 0
            while got < sz:
                got += os.preadv(fd, [mv[got:]], off + got)
            bufs.append(b)
        return bufs
    finally:
        os.close(fd)


def _read_layer(layout, i):
    """Bytes for layer i's K and V from every segment (runs on an I/O thread)."""
    res = []
    for path, ranges, _ in layout:
        spans = ranges[2 * i] + ranges[2 * i + 1]
        res.append((spans, _pread_all(path, spans)))
    return res


def _to_tensor(spans, bufs, codec, dtype):
    if codec == "q8":
        w, s, b = (mx.array(buf).view(dt).reshape(shp) for (_, _, shp, dt), buf in zip(spans, bufs))
        return mx.dequantize(w, s, b, group_size=64, bits=8).astype(dtype)
    (_, _, shp, dt), = spans
    return mx.array(bufs[0]).view(dt).reshape(shp)


def _build_layer(parts, codec, dtype):
    ks, vs = [], []
    for spans, bufs in parts:
        h = len(spans) // 2
        ks.append(_to_tensor(spans[:h], bufs[:h], codec, dtype))
        vs.append(_to_tensor(spans[h:], bufs[h:], codec, dtype))
    cat = lambda xs: xs[0] if len(xs) == 1 else mx.concatenate(xs, axis=2)
    return cat(ks), cat(vs)


def _logits(model, h):
    h = model.model.norm(h)
    if getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def load_then_compute(model, store, key, tokens):
    """Baseline with identical I/O code: read every layer, then run the forward pass."""
    layout = _layout(store, key)
    n_layers = len(model.model.layers)
    futs = [_pool.submit(_read_layer, layout, i) for i in range(n_layers)]
    caches = []
    for f in futs:
        c = KVCache()
        c.keys, c.values = _build_layer(f.result(), store.codec, layout[0][2])
        c.offset = c.keys.shape[2]
        caches.append(c)
    mx.eval([x for c in caches for x in (c.keys, c.values)])
    logits = model(mx.array(tokens)[None], cache=caches)
    return caches, logits[:, -1, :]


def check_supported(model):
    """Streaming needs a plain decoder stack (embed -> layers -> norm) where every layer
    uses a standard full-attention KVCache. Tested: Llama 3.x, Qwen3."""
    inner = getattr(model, "model", None)
    missing = [a for a in ("embed_tokens", "layers", "norm") if not hasattr(inner, a)]
    if missing:
        raise NotImplementedError(f"{type(model).__module__}: no model.model.{'/'.join(missing)}")
    if getattr(inner, "swa_idx", None) is not None:
        raise NotImplementedError("sliding-window attention layers are not supported yet")
    if hasattr(model, "make_cache"):
        kinds = {type(c).__name__ for c in model.make_cache()}
        if kinds != {"KVCache"}:
            raise NotImplementedError(f"cache types {sorted(kinds)} are not supported yet (only KVCache)")


def stream_prefill(model, store, key, tokens):
    """Resume a session: run the new tokens through the model while KV streams in per layer."""
    check_supported(model)
    layout = _layout(store, key)
    inner = model.model
    n_layers = len(inner.layers)
    futs = [_pool.submit(_read_layer, layout, i) for i in range(n_layers)]  # I/O starts now
    n_ctx = store.sessions[key]["saved"]
    caches = []
    for _ in range(n_layers):
        c = KVCache(); c.offset = n_ctx; caches.append(c)
    h = inner.embed_tokens(mx.array(tokens)[None])
    mask = create_attention_mask(h, caches[0])
    for layer, c, f in zip(inner.layers, caches, futs):
        c.keys, c.values = _build_layer(f.result(), store.codec, layout[0][2])
        h = layer(h, mask, cache=c)
        mx.async_eval(h)  # GPU runs this layer while we wait for the next layer's bytes
    return caches, _logits(model, h)[:, -1, :]
