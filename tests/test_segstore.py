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


def _disk(path):
    return sum(f.stat().st_size for f in path.iterdir() if f.suffix != ".json")


def _branch(prompt_cache, n, seed):
    """Copy a cache (sharing nothing) and append n tokens, as a chat continuing a prompt."""
    c = []
    for p in prompt_cache:
        x = KVCache()
        x.update_and_fetch(p.keys[..., :p.offset, :], p.values[..., :p.offset, :])
        c.append(x)
    return grow(c, n, seed)


def test_fork_shares_the_prompt_on_disk(tmp_path):
    store = SegmentKVStore(str(tmp_path))
    prompt = grow([KVCache() for _ in range(LAYERS)], 1000, 10)
    store.save("doc", prompt)
    prompt_bytes = _disk(tmp_path)
    chats = []
    for i in range(3):
        store.fork("doc", f"chat{i}")
        c = _branch(prompt, 50, 20 + i)
        store.save(f"chat{i}", c)
        chats.append(c)
    per_token = prompt_bytes / 1000
    assert _disk(tmp_path) == prompt_bytes + 3 * 50 * per_token  # prompt stored once
    for i, c in enumerate(chats):
        for (k, v), r in zip(live(c), store.load(f"chat{i}")):
            assert mx.array_equal(k, r.keys).item() and mx.array_equal(v, r.values).item()


def test_rewrite_and_delete_keep_shared_prompt(tmp_path):
    store = SegmentKVStore(str(tmp_path), append=False)  # full-rewrite mode
    prompt = grow([KVCache() for _ in range(LAYERS)], 300, 30)
    store.save("doc", prompt)
    store.fork("doc", "chat")
    store.save("chat", _branch(prompt, 40, 31))  # rewrite would have deleted doc's file
    store.delete("chat")
    for (k, _), r in zip(live(prompt), store.load("doc")):
        assert mx.array_equal(k, r.keys).item()
    store.delete("doc")
    assert _disk(tmp_path) == 0


@pytest.mark.parametrize("codec", ["raw", "lz", "q8"])
def test_index_survives_restart(tmp_path, codec):
    store = SegmentKVStore(str(tmp_path), codec=codec)
    cache = grow([KVCache() for _ in range(LAYERS)], 200, 40, mx.bfloat16)
    store.save("s", cache)
    grow(cache, 30, 41, mx.bfloat16)
    store.save("s", cache)
    store.save_index()
    del store
    reopened = SegmentKVStore.open(str(tmp_path))  # e.g. a new process
    restored = reopened.load("s")
    assert restored[0].offset == 230 and restored[0].keys.dtype == mx.bfloat16
    for (k, _), r in zip(live(cache), restored):
        ok = mx.array_equal(k, r.keys).item() if codec != "q8" else mx.abs(k - r.keys).max().item() < 0.05
        assert ok


def test_compact_merges_segments(tmp_path):
    store = SegmentKVStore(str(tmp_path))
    cache = [KVCache() for _ in range(LAYERS)]
    for step in range(10):
        grow(cache, 17 + step, 50 + step, mx.bfloat16)
        store.save("c", cache)
    assert store.compact("c") == 10
    assert len(store.sessions["c"]["segs"]) == 1 and len(list(tmp_path.iterdir())) == 1
    for (k, v), r in zip(live(cache), store.load("c")):
        assert mx.array_equal(k, r.keys).item() and mx.array_equal(v, r.values).item()
    grow(cache, 5, 99, mx.bfloat16)  # appending after compaction still works
    store.save("c", cache)
    assert store.load("c")[0].offset == cache[0].offset


def test_compact_keeps_shared_prompt_shared(tmp_path):
    store = SegmentKVStore(str(tmp_path))
    prompt = grow([KVCache() for _ in range(LAYERS)], 400, 60)
    store.save("doc", prompt)
    store.fork("doc", "chat")
    chat = _branch(prompt, 30, 61)
    for step in range(4):
        grow(chat, 10, 62 + step)
        store.save("chat", chat)
    assert store.compact("chat") == 4
    assert store.sessions["chat"]["segs"][0] == store.sessions["doc"]["segs"][0]  # still shared
    for (k, _), r in zip(live(chat), store.load("chat")):
        assert mx.array_equal(k, r.keys).item()
