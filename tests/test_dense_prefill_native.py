"""Learner native attention: dispatch, online-state transitions and layout edges."""

import mlx.core as mx
import pytest

import tiny_llm.week2_kernels as kernels
from tiny_llm.qwen3_week2 import Qwen3ModelWeek2
from .utils import assert_allclose, tiny_qwen3_mlx_model


def fixture(shape, phase):
    count = 1
    for dim in shape:
        count *= dim
    return mx.sin(mx.arange(count, dtype=mx.float32) * 0.031 + phase).reshape(shape).astype(mx.bfloat16)


@pytest.fixture(autouse=True)
def require_native(monkeypatch):
    assert kernels.DENSE_PREFILL_MMA_IMPLEMENTED

    def reject_readable(*args, **kwargs):
        raise AssertionError("operator unexpectedly used readable compatibility")

    monkeypatch.setattr(kernels, "scaled_dot_product_attention", reject_readable)


@pytest.mark.parametrize("length,context,kv_heads,kind", [
    (9, 1, 4, "none"), (15, 16, 2, "causal"),
    (31, 32, 1, "additive"), (32, 33, 2, "additive"),
    (33, 47, 1, "causal"), (65, 79, 2, "none"),
])
def test_native_batch_gqa_and_tile_edges(length, context, kv_heads, kind):
    q = fixture((2, 4, length, 128), 0.1)
    k = fixture((2, kv_heads, context, 128), 0.7)
    v = fixture(k.shape, 1.3)
    mask = None
    if kind == "causal":
        mask = "causal"
    elif kind == "additive":
        # Distinct batch/head biases exercise the flattened mask indexing.
        mask = mx.cos(mx.arange(2 * 4 * context, dtype=mx.float32)).reshape(2, 4, 1, context)
    actual = kernels.dense_prefill_attention_mma(q, k, v, 128**-0.5, mask)
    expected = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=128**-0.5, mask=mask,
    ).astype(mx.bfloat16)
    assert_allclose(actual, expected, mx.bfloat16, atol=0.02, rtol=0.02)


def test_native_rescales_history_and_handles_empty_tiles_per_row():
    q = mx.ones((1, 2, 9, 128), dtype=mx.bfloat16)
    # Later tiles increase the maximum: forgetting to rescale either a or l fails.
    levels = (mx.arange(47) // 16).astype(mx.bfloat16) / 4
    k = mx.contiguous(mx.broadcast_to(levels[None, None, :, None], (1, 1, 47, 128)))
    v = fixture(k.shape, 0.3)
    mask = mx.zeros((9, 47), dtype=mx.float32)
    mask[0, :] = -mx.inf        # no mass in the entire row
    mask[1, :16] = -mx.inf      # first tile empty, later tiles valid
    mask[2, 16:32] = -mx.inf    # empty tile between valid tiles
    mask[3, 16:] = -mx.inf      # preserve old state through empty later tiles
    actual = kernels.dense_prefill_attention_mma(q, k, v, 128**-0.5, mask)
    expected = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=128**-0.5, mask=mask,
    ).astype(mx.bfloat16)
    expected = mx.where(mx.all(mask == -mx.inf, axis=-1, keepdims=True), 0, expected)
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert_allclose(actual, expected, mx.bfloat16, atol=0.02, rtol=0.02)
    assert float(mx.max(mx.abs(actual[:, :, 0])).item()) == 0


def test_native_and_decode_dispatch_share_request_cache():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(head_dim=128), checkpoint="tiled-prefill")
    cache = model.create_kv_cache(capacity=11)
    mx.eval(model(mx.array([list(range(1, 11))]), 0, cache))
    mx.eval(model(mx.array([[11]]), 10, cache))
    for layer in model.layers_inner:
        counts = layer.self_attn.attention_dispatch_counts
        assert counts["tiled_prefill"] == 1
        assert counts["readable"] == counts["tiled_prefill_fallback"] == 1
        assert counts["readable_compatibility"] == 0


@pytest.mark.parametrize("heads,kv_heads", [(0, 1), (4, 0), (3, 2)])
def test_private_entry_rejects_bad_heads_before_division(heads, kv_heads):
    q = fixture((4, 9, 128), 0.1)
    k = fixture((2, 17, 128), 0.7)
    with pytest.raises(RuntimeError, match="GQA head counts"):
        kernels.tiny_llm_ext_private._dense_attention_prefill_mma(
            q, k, k, mx.zeros((1,)), 128**-0.5, False, False, heads, kv_heads,
        )
