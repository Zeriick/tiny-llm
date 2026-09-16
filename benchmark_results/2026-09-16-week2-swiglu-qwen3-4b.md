# Week 2 Day 4: SwiGLU checkpoint

Recorded on 2026-09-16 for the third fused-operator checkpoint: RMSNorm, RoPE,
and SwiGLU are enabled together with the earlier Week 2 optimizations.

## Environment

- Chip: Apple M4; unified memory: 24 GiB.
- macOS: 26.6.2; architecture: arm64.
- Python: 3.12.12; MLX: 0.32.0; mlx-lm: 0.31.3.
- Model: `Qwen/Qwen3-4B-MLX-4bit`.
- Cached model revision: `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`.
- Implementation: the current local `tiny_llm` working tree, including the
  custom Week 2 RMSNorm, RoPE, and SwiGLU extensions.

## Command and workload

```bash
PDM_LOG_DIR=/tmp/pdm-logs HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-4b \
  --json-output benchmark_results/2026-09-16-week2-swiglu-qwen3-4b.json
```

This uses the requested default workload: GPU, 16 sequential requests,
64–256 input tokens and 64–256 generated tokens per request, seed 0,
one warmup request, and `prefill_logits=all`. Model files are loaded from the
local Hugging Face cache. The JSON preserves the exact synthetic request trace.

The adjacent `.log` file contains the full console output. The benchmark's
JSON configuration currently omits `week2_checkpoint`; the command above and
the log's `Week2Checkpoint=swiglu` header identify the measured checkpoint.

## Result

The command completed successfully (exit status 0).

| Metric | Value |
| --- | ---: |
| Requests | 16 |
| Prompt tokens | 2,966 |
| Generated tokens | 2,595 |
| Measured elapsed time | 154.26 s |
| Output throughput, including prefill | 16.82 tok/s |
| Total throughput, prompt plus output | 36.05 tok/s |
| Prefill throughput | 44.40 tok/s |
| Decode throughput | 29.49 tok/s |

Raw metrics and workload: [JSON](2026-09-16-week2-swiglu-qwen3-4b.json).
The measured elapsed time excludes model loading and the warmup run.

## Correctness checks

- `pdm run build-ext`: passed.
- `pdm run test --week 2 --day 4 -- -k swiglu`: 2 passed.
- Full W2D4 plus RoPE/SwiGLU backend tests: 139 passed, 1 failed.
  Command:
  `pdm run test --week 2 --day 4 -- tests/test_week_2_swiglu_kernels.py tests/test_week_2_rope_kernels.py`.
- All 42 new SwiGLU cases passed, including CPU/GPU, float32/float16/bfloat16,
  scalar and empty inputs, non-multiple threadgroup lengths, non-contiguous
  inputs, parameter validation, and a two-layer prefill/decode model run.
- The failure is `test_week1_keeps_readable_kernels`: Week 1 attention stores
  `q_norm` as an MLX array, while the test expects an `RMSNorm` instance.
  `git show HEAD:src/tiny_llm/qwen3_week1.py` confirms that this representation
  predates the SwiGLU changes; that file has no working-tree changes.

## Measurement scope

This is one complete-model checkpoint run, not an isolated kernel benchmark.
No same-machine before/after speedup is established by this single run.

A subsequent [matched 128-input/65-output comparison](2026-09-16-week2-fused-matched-128-65.md)
records the four cumulative checkpoints under identical settings.
