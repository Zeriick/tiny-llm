"""Behavior at the boundary between the learner code and the new curriculum."""

import mlx.core as mx
import pytest

from tiny_llm.generate import simple_generate_with_kv_cache
from tiny_llm.kv_cache import TinyKvFullCache


def test_capacity_reset_hides_old_history_and_preserves_independent_dtypes():
    cache = TinyKvFullCache(capacity=4)
    keys = mx.ones((1, 1, 3, 2), dtype=mx.float32)
    values = mx.full(keys.shape, 2, dtype=mx.bfloat16)
    cache.update_and_fetch(keys, values)
    cache.materialize()
    cache.reset()
    k, v, offset, _ = cache.update_and_fetch(keys[:, :, :1] * 3, values[:, :, :1])
    assert offset == 1
    assert k.shape == v.shape == (1, 1, 1, 2)
    assert k.dtype == mx.float32 and v.dtype == mx.bfloat16
    assert k.tolist() == [[[[3.0, 3.0]]]]
    assert cache.key_values[0].shape[2] == 4
    assert cache.physical_growth_copy_bytes == 0


@pytest.mark.parametrize("fail", [False, True])
def test_generation_binds_capacity_and_releases_on_success_or_failure(fail):
    class Cache(TinyKvFullCache):
        released = False

        def release(self):
            self.released = True

    class Detokenizer:
        text = ""
        last_segment = ""

        def reset(self):
            pass

        def finalize(self):
            pass

    class Tokenizer:
        eos_token_id = 0
        detokenizer = Detokenizer()

        def encode(self, prompt, add_special_tokens=False):
            return [1, 2]

    class Model:
        use_bounded_kv_capacity = True

        def create_kv_cache(self, capacity=None):
            self.cache = Cache(capacity)
            return [self.cache]

        def __call__(self, tokens, offset, cache, logits_to_keep):
            assert logits_to_keep == 1
            assert offset == 0
            assert cache[0].capacity == 5
            if fail:
                raise RuntimeError("injected model failure")
            return mx.array([[[1.0, 0.0]]])

    model = Model()
    if fail:
        with pytest.raises(RuntimeError, match="injected model failure"):
            simple_generate_with_kv_cache(model, Tokenizer(), "prompt", max_tokens=3)
    else:
        assert simple_generate_with_kv_cache(model, Tokenizer(), "prompt", max_tokens=3) == ""
    assert model.cache.released
