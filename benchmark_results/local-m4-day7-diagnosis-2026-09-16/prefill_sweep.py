"""Paired prefill length sweep; does not change extension dispatch policy."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import mlx.core as mx
import numpy as np
from mlx_lm import load
from benches import profile_week2_kernels as profile


def compare(builders, warmup, iterations, first_reverse):
    names = list(builders)
    samples = {name: [] for name in names}
    orders = []
    for index in range(warmup + iterations):
        order = names[::-1] if (index + first_reverse) % 2 else names
        if index >= warmup:
            orders.append(order)
        for name in order:
            start = perf_counter()
            outputs, resources = builders[name]()
            mx.eval(*outputs)
            elapsed = (perf_counter() - start) * 1e6
            if index >= warmup:
                samples[name].append(elapsed)
            del outputs, resources
        if index >= warmup and (index - warmup + 1) % 12 == 0:
            print(f"  measured {index - warmup + 1}/{iterations} pairs", flush=True)
    medians = {name: median(values) for name, values in samples.items()}
    return dict(samples_us=samples, medians_us=medians, orders=orders,
                change_percent=100 * (medians[names[1]] / medians[names[0]] - 1))


def full_prefill(model, inputs):
    cache = model.create_kv_cache()
    logits = model(inputs, 0, cache, logits_to_keep=1)
    return [logits], cache


def projections(layer):
    return dict(q=layer.self_attn.wq, k=layer.self_attn.wk, v=layer.self_attn.wv,
                o=layer.self_attn.wo, gate=layer.mlp.w_gate, up=layer.mlp.w_up,
                down=layer.mlp.w_down)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if min(args.tokens) < 9 or args.iterations < 2 or args.iterations % 2 or args.warmup < 0:
        parser.error("tokens must be >=9, iterations positive and even, warmup nonnegative")
    source_paths = [ROOT / "src/extensions/src" / name for name in
                    ("quantized_matmul.cpp", "quantized_matmul.metal", "cooperative_matrix.h")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    cpp = source_paths[0].read_text()
    target = int(re.search(r"target_threadgroups = (\d+)", cpp)[1])
    cap = int(re.search(r"max_splits = (\d+)", cpp)[1])
    implementation = profile.load_implementation("tiny_llm")
    mlx_model, _ = load("Qwen/Qwen3-4B-MLX-4bit")
    models = {name: implementation.model_type(mlx_model, checkpoint=name)
              for name in ("simd-matmul", "split-k")}
    report = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  source=profile.source_metadata(ROOT), source_sha256=hashes,
                  device=mx.device_info(),
                  software={name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")},
                  model="Qwen/Qwen3-4B-MLX-4bit", solution="tiny_llm", dtype="bfloat16",
                  prefill_logits="last", batch_size=1, offset=0, decode_attention=False,
                  warmup_per_pass=args.warmup, iterations_per_pass=args.iterations,
                  passes=2, token_orders=[args.tokens, args.tokens[::-1]],
                  policy=dict(target_threadgroups=target, max_splits=cap),
                  method="Host build + synchronized mx.eval. Fresh KV cache retained through eval. No tokenization or sampling. AB/BA implementation order; forward/reverse token sweeps. Independent projection replay uses the original profiler's last-token LM head.",
                  dispatch={}, validation={}, runs=[], summary=[])
    first = models["split-k"].layers_inner[0]
    for rows in args.tokens:
        dispatch = []
        for name, w in projections(first).items():
            reduction = w.scales.shape[1] * w.group_size
            width = w.weight.shape[0]
            tiles = ((rows + 31) // 32) * ((width + 31) // 32)
            splits = min(cap, max(1, target // tiles), reduction // 128)
            boundaries = [i * (reduction // 128) // splits * 128 for i in range(splits + 1)]
            assert boundaries[0] == 0 and boundaries[-1] == reduction
            assert all(a < b and a % 128 == b % 128 == 0 for a, b in zip(boundaries, boundaries[1:]))
            dispatch.append(dict(projection=name, reduction=reduction, output_width=width,
                                 output_tiles=tiles, splits=splits, boundaries=boundaries))
        report["dispatch"][str(rows)] = dispatch
        print(f"tokens={rows}: " + ", ".join(f"{p['projection']}={p['splits']}" for p in dispatch), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pass_index, token_order in enumerate(report["token_orders"]):
        for rows in token_order:
            print(f"Pass {pass_index + 1}/2, tokens={rows}", flush=True)
            mx.random.seed(rows)
            replays = {name: profile.KernelReplay(implementation, model, "prefill", rows)
                       for name, model in models.items()}
            replays["split-k"].hidden = replays["simd-matmul"].hidden
            inputs = mx.arange(rows, dtype=mx.int32).reshape(1, rows)
            mx.eval(inputs)
            builders = {
                "projections": {name: (lambda r=r: (r.projections(), None)) for name, r in replays.items()},
                "full_prefill": {name: (lambda m=m: full_prefill(m, inputs)) for name, m in models.items()},
            }
            if pass_index == 0:
                arrays, dtype_shapes = {}, {}
                for name, build in builders["full_prefill"].items():
                    outputs, cache = build()
                    mx.eval(*outputs)
                    assert len(cache) == len(models[name].layers_inner)
                    assert all(c.offset == rows for c in cache)
                    assert outputs[0].shape == (1, 1, models[name].vocab_size)
                    dtype_shapes[name] = dict(dtype=str(outputs[0].dtype), shape=outputs[0].shape)
                    arrays[name] = np.array(outputs[0].astype(mx.float32))
                    assert np.isfinite(arrays[name]).all()
                    del outputs, cache
                left, right = arrays.values()
                identical = np.array_equal(left.view(np.uint32), right.view(np.uint32))
                fallback = all(p["splits"] == 1 for p in report["dispatch"][str(rows)])
                if fallback:
                    assert identical, f"Fallback logits differ for {rows} tokens"
                report["validation"][str(rows)] = dict(fallback=fallback, bitwise_equal_logits=identical,
                    max_abs_logit_difference=float(np.max(np.abs(left - right))), outputs=dtype_shapes)
                del arrays, left, right
            phases = list(builders) if pass_index == 0 else list(builders)[::-1]
            for phase in phases:
                print(f" {phase}", flush=True)
                measured = compare(builders[phase], args.warmup, args.iterations, pass_index)
                report["runs"].append(dict(pass_index=pass_index, tokens=rows, phase=phase, **measured))
                print(json.dumps(dict(tokens=rows, phase=phase, medians_us=measured["medians_us"],
                                      change_percent=measured["change_percent"])), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            del builders, replays, inputs
    for rows in args.tokens:
        for phase in ("projections", "full_prefill"):
            runs = [r for r in report["runs"] if r["tokens"] == rows and r["phase"] == phase]
            samples = {name: [x for run in runs for x in run["samples_us"][name]] for name in models}
            medians = {name: median(values) for name, values in samples.items()}
            paired_changes = [100 * (candidate / baseline - 1) for run in runs
                              for baseline, candidate in zip(run["samples_us"]["simd-matmul"],
                                                             run["samples_us"]["split-k"])]
            result = dict(tokens=rows, phase=phase, samples_per_checkpoint=len(next(iter(samples.values()))),
                          medians_us=medians, change_percent=100 * (medians["split-k"] / medians["simd-matmul"] - 1),
                          median_paired_change_percent=median(paired_changes),
                          per_pass_changes=[run["change_percent"] for run in runs])
            if phase == "full_prefill":
                result["tokens_per_second"] = {name: rows * 1e6 / value for name, value in medians.items()}
            report["summary"].append(result)
            print(json.dumps(result), flush=True)
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    report["source_unchanged"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
