"""Fused SwiGLU backend and cumulative model checks."""

import mlx.core as mx
import numpy as np
import pytest

from extensions import tiny_llm_ext
from tiny_llm.qwen3_week2 import Qwen3ModelWeek2
from tiny_llm.week2_kernels import swiglu
from .utils import tiny_qwen3_mlx_model


def reference(gate, up):
    g = np.array(gate.astype(mx.float32)).astype(np.float64)
    u = np.array(up.astype(mx.float32)).astype(np.float64)
    return (g / (1 + np.exp(-g))) * u


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize(
    "dtype,tol", [(mx.float32, 3e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)]
)
@pytest.mark.parametrize("shape", [(), (0,), (1,), (257,), (2, 3, 97)])
def test_swiglu_backends(device, dtype, tol, shape):
    with mx.stream(device):
        mx.random.seed(31)
        gate = (mx.random.normal(shape) * 5).astype(dtype)
        up = mx.random.normal(shape).astype(dtype)
        result = swiglu(gate, up)
        mx.eval(result)
        assert result.shape == shape and result.dtype == dtype
        np.testing.assert_allclose(
            np.array(result.astype(mx.float32)), reference(gate, up), rtol=tol, atol=tol
        )


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
def test_swiglu_strided_and_broadcast_inputs(device):
    with mx.stream(device):
        gate = (
            (mx.arange(70, dtype=mx.float32) / 10 - 3)
            .reshape(2, 5, 7)
            .transpose(0, 2, 1)
        )
        up = mx.broadcast_to(mx.arange(5, dtype=mx.float32), gate.shape)
        result = tiny_llm_ext.swiglu(gate, up)
        mx.eval(result)
        np.testing.assert_allclose(
            np.array(result), reference(gate, up), atol=3e-5, rtol=3e-5
        )


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize(
    "dtype,tol", [(mx.float32, 3e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)]
)
def test_swiglu_large_finite_gates(device, dtype, tol):
    with mx.stream(device):
        gate = mx.array([-100, -20, -1, 0, 1, 20, 100], dtype=dtype)
        up = mx.array([2, -1, 3, 0, -3, 1, 2], dtype=dtype)
        result = swiglu(gate, up)
        mx.eval(result)
        np.testing.assert_allclose(
            np.array(result.astype(mx.float32)), reference(gate, up), atol=tol, rtol=tol
        )


@pytest.mark.parametrize(
    "gate,up",
    [
        (mx.ones((2, 3)), mx.ones((3,))),
        (mx.ones((2, 3), dtype=mx.float32), mx.ones((2, 3), dtype=mx.float16)),
        (mx.ones((2, 3), dtype=mx.int32), mx.ones((2, 3), dtype=mx.int32)),
    ],
)
def test_swiglu_invalid_inputs(gate, up):
    with pytest.raises(RuntimeError, match="swiglu:"):
        swiglu(gate, up)


def test_swiglu_checkpoint_runs_prefill_and_decode(monkeypatch):
    import tiny_llm.qwen3_week2 as model_module

    mx.random.seed(43)
    model = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(num_hidden_layers=2), checkpoint="swiglu"
    )
    calls = []

    def counted_swiglu(gate, up):
        calls.append(gate.shape)
        return swiglu(gate, up)

    monkeypatch.setattr(model_module, "swiglu", counted_swiglu)
    cache = model.create_kv_cache()
    prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), 0, cache, logits_to_keep=1)
    mx.eval(prefill)
    decode = model(mx.array([[4]], dtype=mx.int32), 3, cache, logits_to_keep=1)
    mx.eval(decode)
    assert prefill.shape == decode.shape == (1, 1, 128)
    assert mx.all(mx.isfinite(prefill)).item() and mx.all(mx.isfinite(decode)).item()
    assert len(calls) == 4  # Two layers, for both prefill and decode.
    assert all(layer_cache.offset == 4 for layer_cache in cache)
