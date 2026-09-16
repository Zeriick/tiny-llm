"""Read-only paired diagnosis of the repository's current split-K dispatch."""
import json
import hashlib
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import mlx.core as mx
from mlx_lm import load
from benches import profile_week2_kernels as profile
from tiny_llm.quantize import quantized_linear

OUT = Path(sys.argv[1])
assert not OUT.exists(), OUT
mx.random.seed(0)
implementation = profile.load_implementation('tiny_llm')
mlx_model, _ = load('Qwen/Qwen3-4B-MLX-4bit')
models = {name: implementation.model_type(mlx_model, checkpoint=name)
          for name in ('simd-matmul', 'split-k')}
replays = {name: profile.KernelReplay(implementation, model, 'prefill', 32)
           for name, model in models.items()}
replays['split-k'].hidden = replays['simd-matmul'].hidden

def compare(builders, iterations=48, warmup=6):
    names = list(builders)
    samples = {name: [] for name in names}
    orders = []
    for i in range(warmup + iterations):
        order = names if i % 2 == 0 else list(reversed(names))
        if i >= warmup:
            orders.append(order[:])
        for name in order:
            start = perf_counter()
            outputs = builders[name]()
            mx.eval(*outputs)
            elapsed = (perf_counter() - start) * 1e6
            if i >= warmup:
                samples[name].append(elapsed)
            del outputs
    medians = {name: median(values) for name, values in samples.items()}
    result = dict(medians_us=medians, samples_us=samples, orders=orders,
                  warmup=warmup, iterations=iterations)
    result['latency_change_percent'] = 100 * (medians[names[1]] / medians[names[0]] - 1)
    result['block_medians_us'] = [
        {name: median(values[start:start + 12]) for name, values in samples.items()}
        for start in range(0, iterations, 12)]
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('samples_us', 'orders')}), flush=True)
    return result

def projection_weights(layer):
    return dict(q=layer.self_attn.wq, k=layer.self_attn.wk,
                v=layer.self_attn.wv, o=layer.self_attn.wo,
                gate=layer.mlp.w_gate, up=layer.mlp.w_up,
                down=layer.mlp.w_down)

report = dict(captured_at=datetime.now(timezone.utc).isoformat(),
              source=profile.source_metadata(ROOT), device=mx.device_info(),
              model='Qwen/Qwen3-4B-MLX-4bit', rows=32,
              layers=len(models['split-k'].layers_inner),
              note='Host wall-clock build + mx.eval; AB/BA alternation. Batched projections use distinct real weights from all layers; no repeated identical lazy node.',
              dispatch=[], batched_projections={})
source = (ROOT / 'src/extensions/src/quantized_matmul.cpp').read_text()
target = int(re.search(r'target_threadgroups = (\d+)', source)[1])
cap = int(re.search(r'max_splits = (\d+)', source)[1])
report['policy'] = dict(target_threadgroups=target, max_splits=cap)
report['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (ROOT / 'src/extensions/src/quantized_matmul.cpp',
                 ROOT / 'src/extensions/src/quantized_matmul.metal',
                 ROOT / 'src/extensions/src/cooperative_matrix.h')}
for projection, weight in projection_weights(models['split-k'].layers_inner[0]).items():
    reduction = weight.scales.shape[1] * weight.group_size
    width = weight.weight.shape[0]
    output_tiles = (width + 31) // 32
    splits = min(cap, max(1, target // output_tiles), reduction // 128)
    dispatch = dict(projection=projection, reduction=reduction, output_width=width,
                    output_tiles=output_tiles, splits=splits,
                    group_boundaries=[i * (reduction // 128) // splits * 128
                                      for i in range(splits + 1)])
    report['dispatch'].append(dispatch)
    print(json.dumps(dispatch), flush=True)
    inputs = mx.random.normal((32, reduction)).astype(mx.bfloat16)
    mx.eval(inputs)
    weights = {name: [projection_weights(layer)[projection] for layer in model.layers_inner]
               for name, model in models.items()}
    builders = {name: (lambda ws=ws, x=inputs: [quantized_linear(x, w) for w in ws])
                for name, ws in weights.items()}
    report['batched_projections'][projection] = compare(builders)

print('Full projection replay, matching original profiler (last-token LM head):', flush=True)
report['full_projections'] = compare({name: replay.projections for name, replay in replays.items()})
report['observed_dispatch_flags'] = {}
report['finite_outputs'] = {}
for name, replay in replays.items():
    records = []
    original = profile.project
    def record(impl, x, w):
        records.append(dict(input_shape=x.shape, packed_shape=w.weight.shape,
                            simd=w.use_simdgroup_matmul, split=w.use_split_k_matmul))
        return original(impl, x, w)
    profile.project = record
    try:
        outputs = replay.projections()
    finally:
        profile.project = original
    mx.eval(*outputs)
    report['observed_dispatch_flags'][name] = [dict(shape=key, count=count)
        for key, count in Counter(json.dumps(r, sort_keys=True) for r in records).items()]
    report['finite_outputs'][name] = [mx.all(mx.isfinite(out)).item() for out in outputs]
    del outputs
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(f'Wrote {OUT}', flush=True)
