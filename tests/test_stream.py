"""Layer-streamed resume against real models. Downloads models, so opt-in:

  PURGEKV_MODEL_TESTS=1 pytest tests/test_stream.py
"""
import copy, os
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PURGEKV_MODEL_TESTS"), reason="set PURGEKV_MODEL_TESTS=1")

SUPPORTED = ["mlx-community/Qwen3-0.6B-4bit", "mlx-community/Llama-3.2-1B-Instruct-4bit"]
UNSUPPORTED = ["mlx-community/gemma-3-1b-it-4bit", "mlx-community/Qwen3.5-0.8B-4bit"]


@pytest.mark.parametrize("model_id", SUPPORTED)
@pytest.mark.parametrize("codec,reader", [("raw", "legacy"), ("raw", "single"), ("raw", "native"), ("q8", "legacy")])
def test_streamed_matches_load_then_compute(tmp_path, model_id, codec, reader):
    from mlx_lm import load
    from purgekv.common import real_text_tokens, prefill, greedy
    from purgekv.segstore import SegmentKVStore
    from purgekv.stream import stream_prefill, load_then_compute

    model, tok = load(model_id)
    store = SegmentKVStore(str(tmp_path), codec=codec)
    cache, _ = prefill(model, real_text_tokens(tok, 1500, 1))
    store.save("c", cache)
    cache, _ = prefill(model, real_text_tokens(tok, 200, 2), cache)
    store.save("c", cache)
    new = real_text_tokens(tok, 24, 3)
    # greedy() advances the cache it decodes with, so decode each cache exactly once
    a, la = load_then_compute(model, store, "c", new)  # legacy reader as the reference
    out_loaded = greedy(model, a, la, 12)
    b, lb = stream_prefill(model, store, "c", new, reader=reader)
    out_streamed = greedy(model, b, lb, 12)
    assert out_streamed == out_loaded
    if codec == "raw":  # lossless store: also identical to never leaving memory
        ref, lr = prefill(model, new, copy.deepcopy(cache))
        assert out_streamed == greedy(model, ref, lr, 12)


@pytest.mark.parametrize("model_id", UNSUPPORTED)
def test_unsupported_models_are_refused(model_id):
    from mlx_lm import load
    from purgekv.stream import check_supported

    model, _ = load(model_id)
    with pytest.raises(NotImplementedError):
        check_supported(model)


@pytest.mark.parametrize("model_id", SUPPORTED)
def test_prefetched_resume_is_identical(tmp_path, model_id):
    from mlx_lm import load
    from purgekv.common import real_text_tokens, prefill, greedy
    from purgekv.segstore import SegmentKVStore
    from purgekv.stream import stream_prefill, prefetch

    model, tok = load(model_id)
    store = SegmentKVStore(str(tmp_path))
    cache, _ = prefill(model, real_text_tokens(tok, 1200, 1))
    store.save("c", cache)
    new = real_text_tokens(tok, 24, 3)
    p = prefetch(model, store, "c")  # e.g. when the user opens the chat
    b, lb = stream_prefill(model, store, "c", new, prefetched=p)
    ref, lr = prefill(model, new, copy.deepcopy(cache))
    assert greedy(model, b, lb, 12) == greedy(model, ref, lr, 12)
