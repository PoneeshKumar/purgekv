"""SegmentKVStore and the lossless codec, on synthetic caches (no model download)."""
import numpy as np
import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from purgekv.segstore import SegmentKVStore, lz_pack, lz_unpack

LAYERS, HEADS, DIM = 4, 2, 64


def grow(cache, n, seed, dtype=mx.float16):
    """Append n tokens of random K/V to every layer, the way a forward pass would."""
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(dtype)
    for c in cache:
        c.update_and_fetch(k, k * 0.5)
    return cache


def live(cache):
    return [(c.keys[..., :c.offset, :], c.values[..., :c.offset, :]) for c in cache]


@pytest.mark.parametrize("codec", ["raw", "lz"])
def test_lossless_round_trip_across_appends(tmp_path, codec):
    store = SegmentKVStore(str(tmp_path), codec=codec)
    cache = grow([KVCache() for _ in range(LAYERS)], 300, 1)
    store.save("s", cache)
    grow(cache, 77, 2)
    store.save("s", cache)
    restored = store.load("s")
    assert restored[0].offset == 377
    for (k, v), r in zip(live(cache), restored):
        assert mx.array_equal(k, r.keys).item() and mx.array_equal(v, r.values).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])  # bf16 used to crash in numpy
def test_q8_is_close(tmp_path, dtype):
    store = SegmentKVStore(str(tmp_path), codec="q8")
    cache = grow([KVCache() for _ in range(LAYERS)], 256, 3, dtype)
    store.save("s", cache)
    for (k, _), r in zip(live(cache), store.load("s")):
        assert mx.abs(k - r.keys).max().item() < 0.05


def test_append_writes_each_token_once(tmp_path):
    full = SegmentKVStore(str(tmp_path / "full"), append=False)
    app = SegmentKVStore(str(tmp_path / "app"), append=True)
    cache = [KVCache() for _ in range(LAYERS)]
    for step in range(5):
        grow(cache, 100, step)
        full.save("s", cache)
        app.save("s", cache)
    per_token = LAYERS * 2 * HEADS * DIM * 2
    assert app.bytes_written == 500 * per_token
    assert full.bytes_written == (100 + 200 + 300 + 400 + 500) * per_token


@pytest.mark.parametrize("dtype", [np.float16, "bf16"])
def test_lz_codec_round_trip(dtype):
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(1 << 20).astype(np.float32)
    raw = (vals.astype(np.float16) if dtype is np.float16 else (vals.view(np.uint32) >> 16).astype(np.uint16)).view(np.uint8)
    out = lz_unpack(lz_pack(raw), raw.size)
    assert np.array_equal(out, raw)
