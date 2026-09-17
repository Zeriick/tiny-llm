"""Requires the temporary benchmark-hook.patch; never installs a public operator."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import mlx.core as mx
import numpy as np
from mlx_lm import load
from benches.profile_week2_kernels import load_implementation, source_metadata
from tiny_llm.quantize import dequantize_weights, quantized_matmul

RULES = "TINY_LLM_BENCH_SPLIT_K_RULES"
TRACE = "TINY_LLM_BENCH_SPLIT_K_TRACE"
NAMES = ["s1", "s2", "s4", "s8"]


def weights(layer):
    return dict(q=layer.self_attn.wq, k=layer.self_attn.wk, v=layer.self_attn.wv,
                o=layer.self_attn.wo, gate=layer.mlp.w_gate, up=layer.mlp.w_up,
                down=layer.mlp.w_down)


def select(specification):
    if specification is None:
        os.environ.pop(RULES, None)
    else:
        os.environ[RULES] = specification


@contextmanager
def tracing(path):
    # C++ stderr, not just Python sys.stderr. Never enabled in timed regions.
    with path.open("a") as stream:
        original = os.dup(2)
        os.dup2(stream.fileno(), 2)
        os.environ[TRACE] = "1"
        try:
            yield
        finally:
            os.environ.pop(TRACE, None)
            os.dup2(original, 2)
            os.close(original)


def measure(builders, rules, orders, warmup, iterations):
    samples = {name: [] for name in builders}
    measured_orders = []
    for index in range(warmup + iterations):
        order = orders[index % len(orders)]
        if index >= warmup:
            measured_orders.append(list(order))
        for name in order:
            select(rules[name])  # Previous eval has completed before changing policy.
            start = perf_counter()
            outputs, resources = builders[name]()
            mx.eval(*outputs)
            elapsed = (perf_counter() - start) * 1e6
            if index >= warmup:
                samples[name].append(elapsed)
            del outputs, resources
        if len(builders) > 4 and index >= warmup and (index - warmup + 1) % 6 == 0:
            print(f"  {index - warmup + 1}/{iterations} rounds", flush=True)
    paired = {name: [100 * (value / baseline - 1)
                     for value, baseline in zip(values, samples["s1"])]
              for name, values in samples.items()}
    return dict(samples_us=samples, orders=measured_orders,
                medians_us={name: median(values) for name, values in samples.items()},
                paired_changes_percent=paired,
                median_paired_change_percent={name: median(values) for name, values in paired.items()})


def call(x, w, enabled=True):
    return quantized_matmul(w.scales, w.biases, w.group_size, w.bits, x, w.weight,
                            transpose_b=True, use_simdgroup=True, use_split_k=enabled)


def choose_candidates(report):
    candidates = {}
    for rows in report["tokens"]:
        shape_runs = {}
        for run in report["runs"]:
            if run["tokens"] == rows:
                shape_runs.setdefault((run["reduction"], run["output_width"]), []).append(run)
        selections = []
        for (reduction, width), runs in shape_runs.items():
            scores = {name: median([v for run in runs for v in run["paired_changes_percent"][name]]) for name in NAMES}
            best = min(scores, key=scores.get)
            pass_scores = [median([v for run in runs if run["pass_index"] == p
                                   for v in run["paired_changes_percent"][best]]) for p in (0, 1)]
            # Include small repeatable gains for independent whole-model validation.
            # This exploratory screen is not a claim of statistical significance.
            selected = best if scores[best] < -1.0 and all(v < 0 for v in pass_scores) else "s1"
            selections.append(dict(reduction=reduction, output_width=width,
                                   selected_splits=int(selected[1:]), measured_best=best,
                                   scores_percent=scores, best_per_pass_percent=pass_scores))
        specification = ";".join(f"{rows},{r['reduction']},{r['output_width']},{r['selected_splits']}" for r in selections)
        candidates[str(rows)] = dict(rules=specification + ";0,0,0,1", shapes=selections)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("operators", "prefill"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operators", type=Path)
    parser.add_argument("--tokens", nargs="+", type=int, default=[32, 64, 128])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix(".dispatch.log")
    if trace.exists():
        parser.error("dispatch log exists")
    select(None)
    os.environ.pop(TRACE, None)
    cpp = ROOT / "src/extensions/src/quantized_matmul.cpp"
    assert "TEMPORARY BENCHMARK HOOK" in cpp.read_text()
    implementation = load_implementation("tiny_llm")
    mlx_model, _ = load("Qwen/Qwen3-4B-MLX-4bit")
    model = implementation.model_type(mlx_model, checkpoint="split-k")
    paths = [ROOT / "src/extensions/src" / n for n in ("quantized_matmul.cpp", "quantized_matmul.metal", "cooperative_matrix.h")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(stage=args.stage, captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  source=source_metadata(ROOT), source_sha256=hashes, device=mx.device_info(),
                  software={n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm")},
                  model="Qwen/Qwen3-4B-MLX-4bit", dtype="bfloat16", tokens=args.tokens,
                  method="Synchronized public primitive, including FP32 partial and reduction; same binary for every split count. Policy changed only between completed mx.eval calls. No GPU timestamp claims.",
                  runs=[], validation=[], summary=[])

    if args.stage == "operators":
        report.update(warmup_per_pass=24, iterations_per_pass=48, passes=2,
                      order="All 24 permutations of S=1,2,4,8 repeated twice per pass; reverse case sweep on second pass.",
                      oracle="Existing Python dequantize_weights equation and independent NumPy FP64 einsum on eight output columns; every output checked finite and BF16 shape; S=1 checked bitwise against Day 5.")
        first = weights(model.layers_inner[0])
        cases = [(rows, name) for rows in args.tokens for name in first]
        rules = {name: f"0,0,0,{name[1:]}" for name in NAMES}
        for pass_index, case_order in enumerate((cases, cases[::-1])):
            for rows, projection in case_order:
                w = first[projection]
                reduction, width = w.scales.shape[1] * 128, w.weight.shape[0]
                mx.random.seed(rows + reduction + width)
                x = mx.random.normal((rows, reduction)).astype(mx.bfloat16)
                mx.eval(x, w.scales, w.biases, w.weight)
                if pass_index == 0:
                    columns = sorted(set([0, 1, 7, 31, 32, width // 2, width - 2, width - 1]))
                    indices = mx.array(columns, dtype=mx.int32)
                    unpacked = dequantize_weights(w.weight[indices], w.scales[indices], w.biases[indices], 128, 4)
                    expected = np.einsum("mn,kn->mk", np.array(x.astype(mx.float32)).astype(np.float64),
                                         np.array(unpacked.astype(mx.float32)).astype(np.float64), optimize=False)
                    select(None)
                    day5 = np.array(call(x, w, enabled=False).astype(mx.float32))
                    with tracing(trace):
                        for name in NAMES:
                            select(rules[name])
                            actual = call(x, w)
                            mx.eval(actual)
                            assert actual.dtype == mx.bfloat16 and actual.shape == (rows, width)
                            values = np.array(actual.astype(mx.float32))
                            assert np.isfinite(values).all()
                            np.testing.assert_allclose(values[:, columns], expected, rtol=5e-3, atol=2e-3)
                            if name == "s1":
                                assert np.array_equal(values.view(np.uint32), day5.view(np.uint32))
                            splits = int(name[1:])
                            boundaries = [i * (reduction // 128) // splits * 128 for i in range(splits + 1)]
                            assert boundaries[0] == 0 and boundaries[-1] == reduction
                            assert all(a < b and a % 128 == b % 128 == 0 for a, b in zip(boundaries, boundaries[1:]))
                            report["validation"].append(dict(tokens=rows, projection=projection, split=splits,
                                reduction=reduction, output_width=width, boundaries=boundaries,
                                oracle_max_abs_error=float(np.max(np.abs(values[:, columns] - expected)))))
                            del actual, values
                    del expected, unpacked, day5
                builders = {name: (lambda x=x, w=w: ([call(x, w)], None)) for name in NAMES}
                orders = list(itertools.permutations(NAMES))
                if pass_index:
                    orders.reverse()
                result = measure(builders, rules, orders, 24, 48)
                report["runs"].append(dict(pass_index=pass_index, tokens=rows, projection=projection,
                                           reduction=reduction, output_width=width, **result))
                print(json.dumps(dict(pass_index=pass_index, tokens=rows, projection=projection,
                                      us=result["medians_us"], paired_percent=result["median_paired_change_percent"])), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["candidates"] = choose_candidates(report)
        report["candidate_selection"] = "Best split with median gain >1% and improvement in both passes; exploratory until full-prefill validation. K/V and gate/up pooled by identical shape."
        expected_lines = [f"BENCH_SPLIT_K M={v['tokens']} N={v['reduction']} K={v['output_width']} S={v['split']}" for v in report["validation"]]
        actual_lines = [line for line in trace.read_text().splitlines() if line.startswith("BENCH_SPLIT_K")]
        assert actual_lines == expected_lines, "Built extension did not honor forced split counts"
        report["verified_dispatch_count"] = len(actual_lines)
    else:
        assert args.operators
        operators = json.loads(args.operators.read_text())
        assert operators["source_sha256"] == hashes
        report.update(prefill_logits="last", batch_size=1, warmup_per_pass=6,
                      iterations_per_pass=12, passes=2,
                      order="Six variants rotated through all positions, in forward and reverse order. Forward/reverse length sweep.",
                      candidates=operators["candidates"])
        day5_model = implementation.model_type(mlx_model, checkpoint="simd-matmul")
        names = NAMES + ["auto", "candidate"]
        for pass_index, case_order in enumerate((args.tokens, args.tokens[::-1])):
            for rows in case_order:
                inputs = mx.arange(rows, dtype=mx.int32).reshape(1, rows)
                mx.eval(inputs)
                def build(model=model):
                    cache = model.create_kv_cache()
                    return [model(inputs, 0, cache, logits_to_keep=1)], cache
                rules = {name: f"0,0,0,{name[1:]}" for name in NAMES}
                rules.update(auto=None, candidate=report["candidates"][str(rows)]["rules"])
                if pass_index == 0:
                    select(None)
                    outputs, cache = build(day5_model)
                    day5 = np.array(outputs[0].astype(mx.float32))
                    del outputs, cache
                    for name in names:
                        select(rules[name])
                        with tracing(trace):
                            outputs, cache = build()
                            mx.eval(*outputs)
                        assert outputs[0].dtype == mx.bfloat16
                        assert outputs[0].shape == (1, 1, model.vocab_size)
                        assert all(c.offset == rows for c in cache)
                        values = np.array(outputs[0].astype(mx.float32))
                        assert np.isfinite(values).all()
                        if name == "s1":
                            assert np.array_equal(values.view(np.uint32), day5.view(np.uint32))
                        report["validation"].append(dict(tokens=rows, variant=name,
                            max_abs_logit_difference=float(np.max(np.abs(values - day5))),
                            bitwise_equal=np.array_equal(values.view(np.uint32), day5.view(np.uint32)),
                            same_argmax=bool(np.argmax(values) == np.argmax(day5))))
                        del outputs, cache, values
                orders = []
                for sequence in (names, names[::-1]):
                    orders.extend(sequence[i:] + sequence[:i] for i in range(len(names)))
                if pass_index:
                    orders.reverse()
                print(f"Full prefill pass={pass_index + 1} tokens={rows}, candidate={rules['candidate']}", flush=True)
                result = measure({name: build for name in names}, rules, orders, 6, 12)
                report["runs"].append(dict(pass_index=pass_index, tokens=rows, **result))
                print(json.dumps(dict(tokens=rows, us=result["medians_us"], paired_percent=result["median_paired_change_percent"])), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                del inputs

    for key in sorted({(r["tokens"], r.get("projection", "full_prefill")) for r in report["runs"]}):
        runs = [r for r in report["runs"] if (r["tokens"], r.get("projection", "full_prefill")) == key]
        names = list(runs[0]["samples_us"])
        report["summary"].append(dict(tokens=key[0], case=key[1],
            medians_us={n: median([v for r in runs for v in r["samples_us"][n]]) for n in names},
            paired_percent={n: median([v for r in runs for v in r["paired_changes_percent"][n]]) for n in names},
            per_pass_paired_percent=[r["median_paired_change_percent"] for r in runs]))
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        select(None)
        os.environ.pop(TRACE, None)
