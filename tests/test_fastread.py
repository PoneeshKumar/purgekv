"""Single-copy readers (Python and C++) against the legacy path, no model needed."""
import os
import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from purgekv import fastread
from purgekv.segstore import SegmentKVStore
from purgekv.stream import _layout

from test_segstore import LAYERS, grow, live


@pytest.fixture
def store(tmp_path):
    s = SegmentKVStore(str(tmp_path))
    cache = [KVCache() for _ in range(LAYERS)]
    for step, n in enumerate([300, 7, 129, 1, 64]):  # uneven segments
        grow(cache, n, step, mx.bfloat16)
        s.save("c", cache)
    return s, cache


@pytest.mark.parametrize("backend", ["single", "native"])
def test_reader_is_byte_exact_with_headroom(store, backend):
    s, cache = store
    layout = _layout(s, "c")
    for i, (k, v) in enumerate(live(cache)):
        kb, vb, n = fastread.read_layer(layout, i, n_new=40, backend=backend)
        assert n == 501 and kb.shape[2] == fastread.capacity(501, 40) >= 541
        rk = mx.array(kb).view(mx.bfloat16)[..., :n, :]
        rv = mx.array(vb).view(mx.bfloat16)[..., :n, :]
        assert mx.array_equal(rk, k).item() and mx.array_equal(rv, v).item()
        assert not kb[..., n:, :].any()  # spare room is zeroed, like mlx-lm's padding


@pytest.mark.parametrize("backend", ["single", "native"])
def test_reader_errors(store, backend):
    s, _ = store
    layout = _layout(s, "c")
    path = layout[1][0]
    with open(path, "r+b") as f:  # truncate one segment
        f.truncate(os.path.getsize(path) // 2)
    with pytest.raises(OSError):
        for i in range(LAYERS):
            fastread.read_layer(layout, i, 8, backend)
    os.remove(layout[0][0])  # and a missing file
    with pytest.raises(OSError):
        fastread.read_layer(layout, 0, 8, backend)


def test_native_matches_python_on_scattered_reads(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2**16, size=(1, 8, 1000, 64), dtype=np.uint16)
    path = str(tmp_path / "seg.bin")
    data.tofile(path)
    jobs = [(path, 0, 0, 1000, data.shape)]
    a = np.zeros((1, 8, 1024, 64), np.uint16)
    b = np.zeros_like(a)
    fastread.read_python(jobs, a)
    fastread.read_native(jobs, b)
    assert np.array_equal(a, b) and np.array_equal(a[:, :, :1000], data)
