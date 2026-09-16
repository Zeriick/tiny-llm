# Matched Week 2 checkpoint comparison: 128 input / 65 output

## Question and workload

The earlier SwiGLU checkpoint record used 16 variable-length requests,
64–256 input/output tokens, and one warmup. The comparison requested here
uses one request, exactly 128 input tokens and 65 output tokens, and two
warmups. Both use `prefill_logits=all` and the same Apple M4 / 24 GiB machine.
The earlier record's 154.26 seconds includes 2,966 prompt tokens and 2,595
generated tokens; it is not a latency measurement for one 128/65 request.

## Matched complete-model measurement

```bash
PDM_LOG_DIR=/tmp/pdm-logs PYTHONUNBUFFERED=1 \
pdm run bench-week2-progression --offline --solution tiny_llm \
  --model qwen3-4b \
  --variant week2-quantized-matvec --variant week2-rmsnorm \
  --variant week2-rope --variant week2-swiglu \
  --input-len 128 --output-len 65 --warmup 2 --repeats 2 \
  --prefill-logits all \
  --json-output benchmark_results/2026-09-16-week2-fused-matched-128-65.json
```

The tool runs each variant in a fresh sequential process, first in forward
order and then in reverse order. Each subprocess uses `--num-seqs 1` and
fixed min/max input and output lengths. All eight runs completed successfully.

| Checkpoint | Prefill tok/s | Decode tok/s | Output tok/s, including prefill |
| --- | ---: | ---: | ---: |
| quantized-matvec | 33.490 | 19.150 | 9.070 |
| rmsnorm | 34.585 | 21.280 | 9.675 |
| rope | 42.070 | 28.690 | 12.325 |
| swiglu | 39.915 | 26.535 | 11.560 |

Values are medians of two samples. In these runs, the cumulative SwiGLU
checkpoint exceeds quantized-matvec by 38.6% in decode throughput and 27.5%
in output throughput. Relative to the RoPE checkpoint, however, it is 7.5%
lower in decode throughput and 6.2% lower in output throughput.

The small sample count and variability limit attribution: quantized-matvec
decode samples were 17.25 and 21.05 tok/s; RMSNorm samples were 17.53 and
25.03 tok/s. The RoPE-to-SwiGLU difference is an observed complete-model
result, not proof that the standalone SwiGLU kernel is slower.

Raw samples, execution order, source revision, and host metadata:
[complete-model JSON](2026-09-16-week2-fused-matched-128-65.json).

## Isolated SwiGLU call measurement

Using the repository's `benchmark_comparison` helper, compare the current
`tiny_llm.basics.silu(gate) * up` against `tiny_llm.week2_kernels.swiglu(gate, up)`.
Inputs are evaluated beforehand. Both variants use the same random BF16
inputs, seed 0, 20 warmup rounds, and 200 measured rounds in alternating order.
Timing includes construction of a fresh operation and `mx.eval` on every call;
these are host-observed call latencies, not GPU-only kernel durations.

| Shape | Readable median | Fused median | Latency reduction |
| --- | ---: | ---: | ---: |
| `[1, 1, 9728]` (decode) | 168.83 us | 155.75 us | 7.7% |
| `[1, 128, 9728]` (prefill) | 386.56 us | 316.17 us | 18.2% |

The intermediate dimension 9,728 is taken from the cached Qwen3-4B model
configuration. Full timing samples and measurement orders are in the
[operator JSON](2026-09-16-swiglu-matched-operator.json).

These operator samples show lower latency for the fused call at both shapes.
They do not resolve the cause of the complete-model RoPE-to-SwiGLU difference;
more complete-model samples and profiling would be needed to attribute it.

## Metric definitions

- Output throughput: all generated tokens divided by measured request time,
  including prefill and loop overhead.
- Decode throughput: generated tokens after the first token divided by decode
  time. For this workload, the first token comes from prefill and the decode
  loop produces 64 tokens.
- Prefill throughput: input tokens divided by prefill time.
- Loading and warmups are excluded from the reported measured request time.
