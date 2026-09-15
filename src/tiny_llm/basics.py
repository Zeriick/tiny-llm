import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    output = x @ w.T

    # 添加偏置（如果提供）
    # bias 形状 (O,) 会自动广播到 (..., O)
    if bias is not None:
        output = output + bias

    return output


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)
