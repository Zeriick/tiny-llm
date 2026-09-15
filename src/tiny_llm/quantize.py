from typing import Any

import mlx.core as mx

from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul
        self.use_mlx_quantized_linear = use_mlx_quantized_linear

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
            use_mlx_quantized_linear=use_mlx_quantized_linear,
        )

# Input:
#   A: M × N (bfloat16 activations)
#   B_quantized: K × (N/8) (uint32, packed weights)
#   scales: K × (N/G) (bfloat16)
#   biases: K × (N/G) (bfloat16)

# Output:
#   C: M × K (bfloat16)

# For each output element C[i, k]:
#   sum = 0  # float accumulator
#   for each group g in 0..(N/G - 1):
#     scale = scales[k, g]
#     bias = biases[k, g]

#     # Process G values in the group (G/8 uint32 packs)
#     for each pack p in 0..(G/8 - 1):
#       packed_value = B_quantized[k, g*(G/8) + p]

#       # Unpack 8 × 4-bit values
#       for bit_offset in [0, 4, 8, 12, 16, 20, 24, 28]:
#         quantized = (packed_value >> bit_offset) & 0xF
#         b_value = quantized * scale + bias
#         a_value = A[i, g*G + p*8 + bit_offset/4]
#         sum = sum + a_value * b_value

#   C[i, k] = bfloat16(sum)

def mlx_quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    pass


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    *leading, reduction = a.shape
    flat_a = a.reshape(-1, reduction)
    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(flat_a),
        mx.contiguous(b),
        transpose_b=transpose_b,
        use_simdgroup=use_simdgroup,
        use_split_k=use_split_k,
    )
    return result.reshape(*leading, -1)


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    if bits <= 0 or 32 % bits != 0:
        raise ValueError("bits must be a positive divisor of 32")

    values_per_word = 32 // bits
    shifts = mx.arange(0, 32, bits, dtype=mx.uint32)
    quantized = (weight[..., None] >> shifts) & ((1 << bits) - 1)
    quantized = quantized.reshape(
        *weight.shape[:-1],
        weight.shape[-1] * values_per_word,
    ).astype(mx.float32)

    expanded_scales = mx.repeat(scales, group_size, axis=-1).astype(mx.float32)
    dequantized = quantized * expanded_scales
    if biases is not None:
        expanded_biases = mx.repeat(biases, group_size, axis=-1).astype(mx.float32)
        dequantized = dequantized + expanded_biases

    return dequantized.astype(scales.dtype)


def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *leading, reduction = a.shape
    flat_a = a.reshape(-1, reduction)
    if flat_a.shape[0] > 8:
        raise ValueError("quantized_matvec_custom supports at most 8 input rows")
    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(flat_a),
        mx.contiguous(b),
        transpose_b=transpose_b,
        use_simdgroup=True,
        use_split_k=False,
    )
    return result.reshape(*leading, -1)


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        use_simdgroup=False,
    )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    rows = 1
    for size in x.shape[:-1]:
        rows *= size
    operation = (
        quantized_matvec_custom
        if rows <= 8 and w.use_simdgroup_matvec
        else quantized_matmul
    )
    kwargs = {}
    if operation is quantized_matmul:
        kwargs["use_simdgroup"] = w.use_simdgroup_matmul
        kwargs["use_split_k"] = w.use_split_k_matmul
    output = operation(
        w.scales,
        w.biases,
        w.group_size,
        w.bits,
        x,
        w.weight,
        True,
        **kwargs,
    )
    return output if bias is None else output + bias
