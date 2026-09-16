"""RoPE backend, head-block boundaries, and model offset reuse checks."""

import mlx.core as mx
import numpy as np
import pytest

from extensions import tiny_llm_ext
from tiny_llm.qwen3_week2 import Qwen3ModelWeek2
from tiny_llm.week2_kernels import FastRoPE, normalize_rope_offsets
from .utils import tiny_qwen3_mlx_model


def rope_reference(x, offsets, dims, base, traditional):
    """Use complex multiplication as an independent rotation reference."""
    data = np.array(x.astype(mx.float32)).astype(np.float64)
    positions = np.asarray(offsets)[:, None] + np.arange(data.shape[1])
    frequencies = np.power(base, -np.arange(dims // 2) / (dims // 2))
    rotation = np.exp(1j * positions[:, :, None, None] * frequencies)
    if traditional:
        pairs = data[..., :dims].reshape(*data.shape[:-1], dims // 2, 2)
        rotated = (pairs[..., 0] + 1j * pairs[..., 1]) * rotation
        pairs[..., 0], pairs[..., 1] = rotated.real, rotated.imag
    else:
        rotated = (data[..., : dims // 2] + 1j * data[..., dims // 2 : dims]) * rotation
        data[..., : dims // 2], data[..., dims // 2 : dims] = rotated.real, rotated.imag
    return data


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize(
    "dtype,tol", [(mx.float32, 3e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)]
)
@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize(
    "heads,head_dim,dims",
    [(1, 2, 2), (3, 16, 16), (4, 16, 16), (5, 19, 10), (9, 33, 2)],
)
def test_rope_backends(device, dtype, tol, traditional, heads, head_dim, dims):
    with mx.stream(device):
        mx.random.seed(17)
        x = mx.random.normal((2, 3, heads, head_dim)).astype(dtype)
        offsets = [-7, 29]
        y = FastRoPE(dims, 64, traditional=traditional)(x, offsets)
        mx.eval(y)
        expected = rope_reference(x, offsets, dims, 10000.0, traditional)
        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)), expected, atol=tol, rtol=tol
        )
        np.testing.assert_array_equal(
            np.array(y[..., dims:].astype(mx.float32)),
            np.array(x[..., dims:].astype(mx.float32)),
        )


@pytest.mark.parametrize(
    "offset", [7, [7, 7], mx.array(7), mx.array([7, 7], dtype=mx.int64)]
)
def test_rope_wrapper_offset_forms(offset):
    x = mx.arange(2 * 3 * 5 * 16, dtype=mx.float32).reshape(2, 3, 5, 16) / 100
    y = FastRoPE(16, 64)(x, offset)
    mx.eval(y)
    np.testing.assert_allclose(
        np.array(y), rope_reference(x, [7, 7], 16, 10000.0, False), atol=3e-5, rtol=3e-5
    )


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
def test_rope_strided_inputs(device):
    with mx.stream(device):
        x = mx.arange(2 * 5 * 3 * 32, dtype=mx.float32).reshape(2, 5, 3, 32)
        x = (x / 100).transpose(0, 2, 1, 3)[..., ::2]
        offsets = mx.array([3, 99, 17, 99], dtype=mx.int32)[::2]
        y = tiny_llm_ext.rope(x, offsets, 10, 10000.0, False)
        mx.eval(y)
        np.testing.assert_allclose(
            np.array(y),
            rope_reference(x, [3, 17], 10, 10000.0, False),
            atol=3e-5,
            rtol=3e-5,
        )


@pytest.mark.parametrize("offsets", [[4096, 32768], [65535, 131071]])
def test_rope_bf16_long_positions(offsets):
    mx.random.seed(23)
    x = mx.random.normal((2, 2, 5, 128)).astype(mx.bfloat16)
    y = FastRoPE(128, 131074, base=1000000)(x, offsets)
    mx.eval(y)
    np.testing.assert_allclose(
        np.array(y.astype(mx.float32)),
        rope_reference(x, offsets, 128, 1000000.0, False),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize("shape", [(0, 3, 5, 16), (2, 0, 5, 16), (2, 3, 0, 16)])
def test_rope_empty_inputs(device, shape):
    with mx.stream(device):
        y = FastRoPE(16, 64)(mx.zeros(shape), 0)
        mx.eval(y)
        assert y.shape == shape


@pytest.mark.parametrize("offset", [[1], mx.array([1]), mx.array([[1], [2]])])
def test_rope_invalid_offsets(offset):
    with pytest.raises(ValueError, match="one offset per batch row"):
        FastRoPE(16, 64)(mx.zeros((2, 3, 5, 16)), offset)


@pytest.mark.parametrize(
    "dims,base",
    [
        (0, 10000),
        (3, 10000),
        (18, 10000),
        (16, 0),
        (16, -1),
        (16, float("nan")),
        (16, float("inf")),
    ],
)
def test_rope_invalid_parameters(dims, base):
    with pytest.raises(RuntimeError, match="rope:"):
        tiny_llm_ext.rope(
            mx.zeros((2, 3, 5, 16)), mx.array([0, 1], mx.int32), dims, base, False
        )


@pytest.mark.parametrize(
    "offset",
    [
        3,
        [3, 7],
        mx.array(3),
        mx.array([3, 7], dtype=mx.int64),
        mx.array([3, 0, 7, 0])[::2],
    ],
)
def test_rope_model_reuses_offsets(offset):
    model = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(num_hidden_layers=2), checkpoint="rope"
    )
    seen = []

    class RecordingRoPE:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __call__(self, x, offset):
            seen.append(offset)
            assert normalize_rope_offsets(offset, x.shape[0]) is offset
            return self.wrapped(x, offset)

    for layer in model.layers_inner:
        layer.self_attn.rope = RecordingRoPE(layer.self_attn.rope)
    cache = model.create_kv_cache()
    # Scalar offset validation still checks the cache's absolute position.
    if isinstance(offset, int):
        for layer_cache in cache:
            layer_cache.offset = offset
    result = model(mx.array([[1], [2]], dtype=mx.int32), offset, cache)
    mx.eval(result)
    assert len(seen) == 4  # Q and K in both layers.
    assert all(value is seen[0] for value in seen)
    assert seen[0].dtype == mx.int32 and seen[0].shape == (2,)
