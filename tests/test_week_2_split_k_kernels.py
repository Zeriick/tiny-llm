"""Public split-K results, group coverage, FP32 intermediates, and exact fallback."""

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm.quantize import dequantize_weights, quantized_matmul


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize(
    "shape,outputs",
    [
        ((9, 384), 1),
        ((17, 2560), 1032),
        ((32, 2560), 1024),
        ((33, 2176), 65),
        ((2, 3, 7, 1152), 35),
    ],
)
def test_split_k_matches_python_equation_gpu(dtype, shape, outputs, monkeypatch):
    with mx.stream(mx.gpu):
        mx.random.seed(sum(shape) + outputs)
        inputs = mx.random.normal(shape).astype(dtype)
        weights = mx.random.normal((outputs, shape[-1])).astype(dtype)
        packed, scales, biases = mx.quantize(weights, group_size=128, bits=4)
        unpacked = dequantize_weights(packed, scales, biases, 128, 4)
        expected = np.einsum(
            "...n,kn->...k",
            np.array(inputs.astype(mx.float32)).astype(np.float64),
            np.array(unpacked.astype(mx.float32)).astype(np.float64),
            optimize=False,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Split-K must use the course-owned extension")

        monkeypatch.setattr(mx, "quantized_matmul", forbidden)
        monkeypatch.setattr(mx, "dequantize", forbidden)
        actual = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
            use_split_k=True,
        )
        assert actual.shape == (*shape[:-1], outputs)
        assert actual.dtype == dtype
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            expected,
            atol=2e-3,
            rtol=5e-3 if dtype == mx.bfloat16 else 6e-4,
        )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("groups", [3, 7, 9, 17, 20])
def test_split_k_covers_each_quantization_group_once_gpu(dtype, groups):
    with mx.stream(mx.gpu):
        rows, outputs = max(9, groups), 35
        # Each row selects all 128 values of exactly one group. Distinct codes,
        # scales, and biases expose missing/overlapping groups or plane strides.
        selected = mx.arange(rows) % groups
        inputs = mx.repeat(selected[:, None] == mx.arange(groups), 128, axis=-1).astype(
            dtype
        )
        column = mx.arange(outputs)[:, None]
        group = mx.arange(groups)[None, :]
        codes = ((column * 3 + group * 5) % 16).astype(mx.uint32)
        packed = mx.repeat(codes * mx.array(0x11111111, mx.uint32), 16, axis=-1)
        scales = ((column + group) % 4 + 1).astype(dtype) / 16
        biases = (((column + group) % 7 - 3) / 32).astype(dtype)
        actual = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
            use_split_k=True,
        )
        expected = (
            (128 * (codes.astype(mx.float32) * scales + biases))
            .T[selected]
            .astype(dtype)
        )
        assert actual.shape == (rows, outputs) and actual.dtype == dtype
        assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("stage", ["partial_store", "reduction"])
def test_split_k_retains_fp32_until_final_output_gpu(dtype, stage):
    with mx.stream(mx.gpu):
        groups = 2 if stage == "partial_store" else 3
        inputs = np.zeros((9, groups * 128), dtype=np.float32)
        inputs[:, 0] = 2048
        if stage == "partial_store":
            # One partial contains 2048.25. Storing it in either BF16 or FP16
            # loses the fractional part before cancellation in the reduction.
            inputs[:, 1] = 0.25
            inputs[:, 128] = -2048
        else:
            # The partials are exactly representable, but their running sum
            # needs FP32 to retain 0.25 between +2048 and -2048.
            inputs[:, 128] = 0.25
            inputs[:, 256] = -2048
        actual = quantized_matmul(
            mx.ones((35, groups), dtype),
            mx.zeros((35, groups), dtype),
            128,
            4,
            mx.array(inputs, dtype),
            mx.full((35, groups * 16), 0x11111111, mx.uint32),
            transpose_b=True,
            use_simdgroup=True,
            use_split_k=True,
        )
        assert mx.array_equal(actual, mx.full((9, 35), 0.25, dtype)).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize(
    "shape,outputs,use_simdgroup",
    [
        ((17, 128), 35, True),  # Only one quantization group.
        ((128, 256), 2560, True),  # Enough output tiles without splitting.
        ((1, 384), 35, True),
        ((8, 384), 35, True),  # Existing matvec schedule.
        ((17, 384), 35, False),
    ],  # Existing scalar schedule.
)
def test_split_k_fallback_is_bitwise_unsplit_gpu(dtype, shape, outputs, use_simdgroup):
    with mx.stream(mx.gpu):
        mx.random.seed(sum(shape) + outputs)
        inputs = mx.random.normal(shape).astype(dtype)
        packed, scales, biases = mx.quantize(
            mx.random.normal((outputs, shape[-1])).astype(dtype),
            group_size=128,
            bits=4,
        )
        unsplit = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=use_simdgroup,
            use_split_k=False,
        )
        requested = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=use_simdgroup,
            use_split_k=True,
        )
        assert requested.shape == unsplit.shape == (shape[0], outputs)
        assert requested.dtype == unsplit.dtype == dtype
        # Byte comparison also distinguishes +0 from -0, unlike array_equal.
        assert (
            np.array(requested.view(mx.uint16)).tobytes()
            == np.array(unsplit.view(mx.uint16)).tobytes()
        )
