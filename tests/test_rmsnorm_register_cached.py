"""Learner GPU RMSNorm: partial lanes, four-value boundary, and wide fallback."""

import mlx.core as mx
import pytest

from tiny_llm.week2_kernels import FastRMSNorm
from .utils import assert_allclose


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("dim", [1, 7, 128, 129, 2560, 4095, 4096, 4097, 8192])
def test_register_cached_and_wide_rmsnorm(dtype, dim):
    with mx.stream(mx.gpu):
        x = mx.sin(mx.arange(6 * dim, dtype=mx.float32) * 0.013).reshape(2, 3, dim).astype(dtype)
        weight = mx.cos(mx.arange(dim, dtype=mx.float32) * 0.019).astype(dtype)
        norm = FastRMSNorm(dim, weight)
        actual = norm(x)
        expected = mx.fast.rms_norm(x, weight, 1e-5)
        mx.eval(actual, expected)
        tolerance = 1e-5 if dtype == mx.float32 else 2e-2
        assert actual.shape == x.shape
        assert actual.dtype == dtype
        assert_allclose(actual, expected, dtype, atol=tolerance, rtol=tolerance)
        path = "register_cached" if dim <= 4096 else "fixed_width_fallback"
        assert norm.dispatch_counts[path] == 1
        assert sum(norm.dispatch_counts.values()) == 1
