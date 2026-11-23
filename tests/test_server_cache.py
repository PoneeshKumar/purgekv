"""SSD tier for mlx_lm.server's prompt cache, on synthetic caches (no model download)."""
import os

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from purgekv.segstore import SegmentKVStore
from purgekv.server import SSDPromptCache

from test_segstore import LAYERS, HEADS, DIM, grow, live

M = "test-model"
GB = 1 << 30


def chat(n, seed, quant=False):
    cache = grow([KVCache() for _ in range(LAYERS)], n, seed, mx.bfloat16)
    if quant:
        cache = [c.to_quantized(group_size=64, bits=8) for c in cache]
    return cache


def same(a, b):
    if isinstance(a[0], QuantizedKVCache):
        return all(mx.array_equal(x[..., :ca.offset, :], y[..., :cb.offset, :]).item()
                   for ca, cb in zip(a, b) for x, y in zip((*ca.keys, *ca.values), (*cb.keys, *cb.values)))
    return all(mx.array_equal(k, k2).item() and mx.array_equal(v, v2).item() for (k, v), (k2, v2) in zip(live(a), live(b)))


def test_quantized_cache_round_trip_is_exact_and_half_size(tmp_path):
    full, q = chat(512, 1), chat(512, 1, quant=True)
    s = SegmentKVStore(str(tmp_path / "q"))
    s.save("c", q[:])
    k = mx.random.normal((1, HEADS, 40, DIM), key=mx.random.key(9)).astype(mx.bfloat16)
    for c in q:
        c.update_and_fetch(k, k)  # an appended turn
    s.save("c", q)
    r = s.load("c")
    assert isinstance(r[0], QuantizedKVCache) and r[0].offset == 552 and same(q, r)
    f = SegmentKVStore(str(tmp_path / "f"))
    f.save("c", full)
    qb = sum(m["nbytes"] for _, m in s.sessions["c"]["segs"]) * 512 / 552
    fb = sum(m["nbytes"] for _, m in f.sessions["c"]["segs"])
    assert 0.5 < qb / fb < 0.56  # 8 bits + a scale and bias per 64 values


def test_quantized_index_survives_restart(tmp_path):
    q = chat(300, 2, quant=True)
    s = SegmentKVStore(str(tmp_path))
    s.save("c", q)
    s.save_index()
    r = SegmentKVStore.open(str(tmp_path)).load("c")
    assert same(q, r)


def test_streaming_and_compaction_refuse_quantized(tmp_path):
    from purgekv.stream import _layout
    s = SegmentKVStore(str(tmp_path))
    q = chat(64, 3, quant=True)
    s.save("c", q)
    with pytest.raises(NotImplementedError):
        _layout(s, "c")
    s.save("c", q)
    with pytest.raises(NotImplementedError):
        s.compact("c")


def toks(n, base=0):
    return list(range(base, base + n))


@pytest.mark.parametrize("quant", [False, True])
def test_evicted_chat_comes_back_from_ssd(tmp_path, quant):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=0)
    a, b = chat(300, 4, quant), chat(200, 5, quant)
    pc.insert_cache(M, toks(301), a)  # cache_key holds one more token than the cache
    pc.insert_cache(M, toks(201, 10_000), b)  # max_size=1: evicts chat a from RAM
    cache, rest = pc.fetch_nearest_cache(M, toks(301) + [7, 8])
    assert rest == [300, 7, 8] and pc.stats["ssd_loads"] == 1 and same(a, cache)
    cache, rest = pc.fetch_nearest_cache(M, toks(201, 10_000) + [1])
    assert pc.stats["ram_hits"] == 1 and rest == [1] and pc.stats["ssd_loads"] == 1  # RAM hit


def test_next_turn_writes_only_new_tokens(tmp_path):
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    a = chat(300, 6)
    pc.insert_cache(M, toks(300), a)
    first = pc.stats["bytes_written"]
    grow(a, 20, 7, mx.bfloat16)
    pc.insert_cache(M, toks(320), a)
    assert pc.stats["bytes_written"] - first == pytest.approx(first * 20 / 300, rel=0.01)
    st = pc.stores[M]
    assert len(st.sessions) == 1  # the 300-token prefix entry was folded into the new one
    r = st.load(next(iter(st.sessions)))
    assert same(a, r)


def test_prefetch_reads_waiting_chat_before_it_returns(tmp_path):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=GB)
    a, b = chat(300, 8), chat(200, 9)
    pc.insert_cache(M, toks(300), a)
    pc.insert_cache(M, toks(200, 10_000), b)  # a leaves RAM -> prefetch starts
    assert list(pc.prefetched) and next(iter(pc.prefetched))[0] == M
    next(iter(pc.prefetched.values()))[0].result()
    cache, rest = pc.fetch_nearest_cache(M, toks(300) + [5])
    assert pc.stats["prefetch_hits"] == 1 and same(a, cache) and rest == [5]


def test_prefetch_respects_budget(tmp_path):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=1)
    pc.insert_cache(M, toks(300), chat(300, 10))
    pc.insert_cache(M, toks(200, 10_000), chat(200, 11))
    assert not pc.prefetched


def test_server_restart_reloads_from_ssd(tmp_path):
    a = chat(300, 12)
    SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0).insert_cache(M, toks(300), a)
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)  # a new process
    cache, rest = pc.fetch_nearest_cache(M, toks(300) + [1, 2])
    assert rest == [1, 2] and same(a, cache)


def test_rewritten_history_reuses_the_shared_start(tmp_path):
    """Qwen3's template drops the empty <think></think> from past replies, so the saved
    chat (prompt + generated reply) is not a prefix of the next request. The SSD tier must
    reuse the shared start and trim the rest, as mlx-lm's RAM cache does."""
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=0)
    a = chat(300, 13)
    saved = toks(250) + [9_001, 9_002] + toks(48, 5_000)   # prompt, think tags, reply
    pc.insert_cache(M, saved, a)
    pc.insert_cache(M, toks(200, 10_000), chat(200, 14))  # push it out of RAM
    nxt = toks(250) + toks(48, 5_000) + [7, 7, 7]            # history without the think tags
    cache, rest = pc.fetch_nearest_cache(M, nxt)
    assert pc.stats["ssd_loads"] == 1 and cache[0].offset == 250 and rest == nxt[250:]
    k, v = live(a)[0]
    assert mx.array_equal(cache[0].keys[..., :250, :], k[..., :250, :]).item()


THINK = [9_001, 9_002]  # Qwen3's empty <think></think>: in the reply, dropped from history later


def agent_chat(pc, turns, rewrite=True):
    """An agent loop: each turn adds a tool result and a reply; another chat then pushes
    this one out of RAM. Returns the last cache and its tokens."""
    hist, nxt = toks(1000), 50_000
    for turn in range(turns):
        prompt = hist + toks(150, nxt)
        reply = toks(50, nxt + 150)
        nxt += 200
        full = prompt + (THINK if rewrite else []) + reply
        cache, _ = pc.fetch_nearest_cache(M, prompt + (THINK if rewrite else []))
        if cache is None:
            cache = [KVCache() for _ in range(LAYERS)]
        grow(cache, len(full) - cache[0].offset, turn, mx.bfloat16)
        pc.insert_cache(M, full, cache)
        pc.insert_cache(M, toks(5, 900_000 + 10 * turn), chat(5, 99))  # another agent's turn
        hist = prompt + reply
    return cache, full


def test_rewritten_history_writes_only_the_last_turn_again(tmp_path):
    """Before 0.4 a Qwen3 chat was written again in full on every turn, and the old
    copies stayed on disk. Now each save rewrites at most the previous turn."""
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=0)
    cache, full = agent_chat(pc, 10)
    st = pc.stores[M]
    per_token = 2 * LAYERS * HEADS * DIM * 2  # keys + values, bf16
    others = 10 * 5 * per_token
    # each turn rewrites the previous one once: about 2x the chat in all, was ~7x (and growing)
    assert pc.stats["bytes_written"] - others < 2.5 * len(full) * per_token
    ours = [k for k, s in st.sessions.items() if s["saved"] > 100]
    assert len(ours) == 1  # the superseded copies were deleted
    assert same(cache, st.load(ours[0]))


def test_exact_history_still_writes_each_token_once(tmp_path):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=0)
    _, full = agent_chat(pc, 10, rewrite=False)
    per_token = 2 * LAYERS * HEADS * DIM * 2
    assert pc.stats["bytes_written"] == (len(full) + 10 * 5) * per_token


def test_a_chat_branching_off_early_keeps_the_original(tmp_path):
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    a = chat(300, 20)
    pc.insert_cache(M, toks(300), a)               # turn 1
    grow(a, 300, 22, mx.bfloat16)
    pc.insert_cache(M, toks(600), a)               # turn 2: forks turn 1
    b = grow(chat(300, 20), 100, 23, mx.bfloat16)  # same first 300 tokens as a
    pc.insert_cache(M, toks(300) + toks(100, 70_000), b)  # an edited second message
    st = pc.stores[M]
    assert sorted(s["saved"] for s in st.sessions.values()) == [400, 600]
    k600 = next(k for k, s in st.sessions.items() if s["saved"] == 600)
    assert same(a, st.load(k600))


def test_a_shorter_chat_does_not_delete_a_longer_one(tmp_path):
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    a = chat(500, 21)
    pc.insert_cache(M, toks(500), a)
    pc.insert_cache(M, toks(480) + [1, 2, 3], chat(483, 21))  # regenerated last reply, shorter
    assert 500 in {s["saved"] for s in pc.stores[M].sessions.values()}


# ---- every other cache layout: sliding window, hybrid recurrent, unknown types ----------
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.cache import ArraysCache, ConcatenateKVCache, RotatingKVCache  # noqa: E402


def mixed(kind):
    """Per-layer caches the way models build them: Gemma 3 (sliding window + full attention),
    Qwen3.5 (linear attention state + full attention), or a type purgekv does not know."""
    other = {"sliding": lambda: RotatingKVCache(max_size=64), "hybrid": lambda: ArraysCache(size=2),
             "unknown": ConcatenateKVCache, "recurrent": lambda: ArraysCache(size=2)}[kind]
    if kind == "recurrent":
        return [other() for _ in range(LAYERS)]
    return [KVCache() if i % 2 else other() for i in range(LAYERS)]


def advance(cache, n, seed):
    """Process n more tokens: attention layers append K/V, recurrent layers replace their state."""
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            c[0] = mx.random.normal((1, 3, 8), key=mx.random.key(seed + i))
            c[1] = mx.random.normal((1, 2, 4, 4), key=mx.random.key(seed + 100 + i))
        else:
            c.update_and_fetch(k, k * 0.5)
    return cache


def same_state(a, b):
    if [type(x) for x in a] != [type(y) for y in b]:
        return False
    for x, y in zip(a, b):
        if type(x) is KVCache:  # its state can include unused preallocated slots
            if x.offset != y.offset or not same([x], [y]):
                return False
            continue
        fx, fy = tree_flatten(x.state), tree_flatten(y.state)
        if len(fx) != len(fy):
            return False
        for (_, u), (_, v) in zip(fx, fy):
            if isinstance(u, mx.array) != isinstance(v, mx.array):
                return False
            if isinstance(u, mx.array) and not (u.shape == v.shape and mx.array_equal(u, v).item()):
                return False
            if not isinstance(u, mx.array) and u != v:
                return False
    return True


@pytest.mark.parametrize("kind", ["sliding", "hybrid", "unknown"])
def test_any_cache_layout_comes_back_from_ssd(tmp_path, kind):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=0)
    a = advance(mixed(kind), 100, 30)  # 100 > the 64-token window: the sliding layers wrapped
    pc.insert_cache(M, toks(100), a)
    pc.insert_cache(M, toks(50, 10_000), advance(mixed(kind), 50, 31))  # evicts a from RAM
    cache, rest = pc.fetch_nearest_cache(M, toks(100) + [7, 8])
    assert pc.stats["ssd_loads"] == 1 and rest == [7, 8] and same_state(a, cache)


def test_untrimmable_chat_is_reused_only_at_its_saved_length(tmp_path):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=0)
    a = advance(mixed("hybrid"), 300, 32)  # a recurrent state cannot be rewound
    pc.insert_cache(M, toks(300), a)
    pc.insert_cache(M, toks(50, 10_000), advance(mixed("hybrid"), 50, 33))
    cache, rest = pc.fetch_nearest_cache(M, toks(250) + [1, 2])  # diverges inside the saved chat
    assert cache is None and pc.stats["ssd_loads"] == 0
    cache, rest = pc.fetch_nearest_cache(M, toks(300) + [1, 2])
    assert pc.stats["ssd_loads"] == 1 and rest == [1, 2] and same_state(a, cache)


def test_untrimmable_checkpoints_are_kept_and_attention_layers_write_deltas(tmp_path):
    """mlx-lm checkpoints at the end of the prompt, then saves prompt + reply. For a model
    that cannot rewind, the checkpoint is what the next request (with the template's rewritten
    reply) reuses, so it must stay; the attention layers still share its files."""
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    a = advance(mixed("hybrid"), 300, 34)
    pc.insert_cache(M, toks(300), a, cache_type="user")
    first = pc.stats["bytes_written"]
    advance(a, 30, 35)
    pc.insert_cache(M, toks(330), a)
    st = pc.stores[M]
    assert sorted(s["saved"] for s in st.sessions.values()) == [300, 330]
    snap = os.path.getsize(os.path.join(st.root, next(s["info"]["snap"] for s in st.sessions.values())))
    kv_first, kv_next = first - snap, pc.stats["bytes_written"] - first - snap
    assert kv_next == pytest.approx(kv_first * 30 / 300, rel=0.01)


def test_mixed_layout_survives_restart(tmp_path):
    a = advance(mixed("sliding"), 120, 36)
    SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0).insert_cache(M, toks(120), a)
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    cache, rest = pc.fetch_nearest_cache(M, toks(120) + [3])
    assert rest == [3] and same_state(a, cache)


def test_mixed_layout_prefetch(tmp_path):
    pc = SSDPromptCache(max_size=1, kv_dir=str(tmp_path), prefetch_bytes=GB)
    a = advance(mixed("hybrid"), 200, 37)
    pc.insert_cache(M, toks(200), a)
    pc.insert_cache(M, toks(50, 10_000), advance(mixed("hybrid"), 50, 38))
    next(iter(pc.prefetched.values()))[0].result()
    cache, rest = pc.fetch_nearest_cache(M, toks(200) + [5])
    assert pc.stats["prefetch_hits"] == 1 and same_state(a, cache)


def test_forgetting_a_chat_removes_its_snapshot(tmp_path):
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    pc.insert_cache(M, toks(100), advance(mixed("sliding"), 30, 39))  # window not full: trimmable
    st = pc.stores[M]
    key = next(iter(st.sessions))
    snap = os.path.join(st.root, st.sessions[key]["info"]["snap"])
    pc._forget(M, key)
    assert not os.path.exists(snap)


def test_pure_recurrent_model_stays_ram_only(tmp_path):
    pc = SSDPromptCache(kv_dir=str(tmp_path), prefetch_bytes=0)
    pc.insert_cache(M, toks(10), advance(mixed("recurrent"), 10, 40))
    assert pc.stats["saves"] == 0  # no layer reports a token count to key the save on
