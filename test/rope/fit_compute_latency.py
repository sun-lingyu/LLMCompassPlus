"""Fit QKNorm/RoPE seconds/output-element from L2-resident Orin/Thor measurements.

Run from the repository root:
    python -m test.rope.fit_compute_latency --device Orin --quick
    python -m test.rope.fit_compute_latency --device Thor
Refit saved measurements without running the hardware:
    python -m test.rope.fit_compute_latency --device Orin --samples PATH
The intercept is fixed to compute_module.launch_latency.rope in device JSON.
Only writes timestamped measurement/fit JSON; hardware configuration is untouched.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hardware_model.device import device_dict
from test.utils import run_remote_command, test_model_dict

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "target_scripts" / "benchmark_rope_l2.py"


def make_shapes(l2_bytes, quick=False, model_name=None, mode="both"):
    """Keep workload row counts unchanged; omit shapes exceeding 80% of L2."""
    shapes = set()
    for name, model in test_model_dict.items():
        if model_name is not None and name != model_name:
            continue
        rows = []
        if mode in ("prefill", "both"):
            rows += (
                [576, 1024] if name == "InternVision" else [512, 768, 1024, 1280, 1536]
            )
        if mode in ("decode", "both"):
            rows += [64, 128]
        for m in rows:
            d = model["head_dim"]
            q = model["num_attention_heads"]
            kv = model["num_key_value_heads"]
            footprint = m * (2 * (q + 2 * kv) * d + 4 * d + 8) + 4 * d + 65536
            shape = (m, d, q, kv)
            if footprint <= 0.8 * l2_bytes:
                shapes.add(shape)
    shapes = sorted(shapes)
    if quick and len(shapes) > 6:
        shapes = [shapes[i] for i in np.linspace(0, len(shapes) - 1, 6, dtype=int)]
    return shapes


def fit_samples(samples, launch_latency):
    x = np.array([s["output_numel"] for s in samples], dtype=float)
    y = np.array([s["latency_seconds"] for s in samples], dtype=float)
    if len(np.unique(x)) < 3:
        return {
            "compute_latency_factor": 0.0,
            "factor_unit": "seconds/output_element",
            "intercept_seconds": float(launch_latency),
            "fit_method": "insufficient_samples",
            "sample_count": len(samples),
            "distinct_output_sizes": len(np.unique(x)),
            "warnings": [
                "Fewer than three distinct L2-resident output sizes; factor is zero."
            ],
        }
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError(
            "Need at least three finite positive measurements at varying sizes"
        )
    if not np.isfinite(launch_latency) or launch_latency < 0:
        raise ValueError("launch_latency must be finite and non-negative")
    # Fixed-intercept least squares. Keep negative (y - intercept) observations:
    # clipping them to zero would bias the fitted slope upward.
    intercept = float(launch_latency)
    factor = float(np.dot(x, y - intercept) / np.dot(x, x))
    predicted = intercept + factor * x
    residual = y - predicted
    total_variance = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(residual**2)) / total_variance if total_variance else None
    by_width = {}
    for n in sorted({s["N"] for s in samples}):
        mask = np.array([s["N"] == n for s in samples])
        by_width[str(n)] = {
            "mean_residual_seconds": float(residual[mask].mean()),
            "mape": float(np.mean(np.abs(residual[mask]) / y[mask])),
        }
    warnings = []
    if factor <= 0:
        warnings.append("Non-positive slope: do not use this fit as a compute factor.")
    if r2 is None or r2 < 0.9:
        warnings.append(
            "Weak linear fit; inspect size/width dependence before using the factor."
        )
    return {
        "compute_latency_factor": factor,
        "factor_unit": "seconds/output_element",
        "intercept_seconds": intercept,
        "intercept_source": "compute_module.launch_latency.rope",
        "fit_method": "fixed_intercept_least_squares",
        "samples_below_intercept": int(np.sum(y < intercept)),
        "r_squared": r2,
        "rmse_seconds": float(np.sqrt(np.mean(residual**2))),
        "mape": float(np.mean(np.abs(residual) / y)),
        "by_qk_width": by_width,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["Orin", "Thor"])
    parser.add_argument(
        "--model", choices=list(test_model_dict), help="Default: all models"
    )
    parser.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true", help="At most six legal shapes for a smoke run"
    )
    mode.add_argument(
        "--samples", type=Path, help="Refit a saved result JSON without measuring"
    )
    args = parser.parse_args()
    if args.samples:
        result = json.loads(args.samples.read_text())
        if result["device"] != args.device or result.get("operator") != "rope":
            parser.error("Saved measurements belong to a different device/operator")
        result["refit_source"] = str(args.samples.resolve())
    else:
        shapes = make_shapes(
            device_dict[args.device].compute_module.l2_size,
            args.quick,
            args.model,
            args.mode,
        )
        source = TARGET.read_text()
        port = 9129 if args.device == "Orin" else 9147
        print(
            f"{args.device}: measuring {len(shapes)} shapes on port {port}", flush=True
        )
        output = (
            run_remote_command(
                "sly",
                "202.120.39.3",
                port,
                [
                    "/home/sly/anaconda3/envs/llmcompass/bin/python",
                    "-B",
                    "-c",
                    source,
                    "--shapes",
                    json.dumps(shapes),
                ],
                work_dir="/home/sly",
            )
            if shapes
            else ""
        )
        samples = [
            json.loads(line[7:])
            for line in output.splitlines()
            if line.startswith("RESULT ")
        ]
        if [
            (s["M"], s["head_dim"], s["num_heads_q"], s["num_heads_kv"])
            for s in samples
        ] != shapes:
            raise ValueError("Incomplete or unexpected measurement results")
        result = {
            "device": args.device,
            "operator": "rope",
            "quick": args.quick,
            "model": args.model,
            "mode": args.mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "l2_fraction_limit": 0.8,
            "samples": samples,
        }
    allowed = set(
        make_shapes(
            device_dict[args.device].compute_module.l2_size,
            model_name=args.model,
            mode=args.mode,
        )
    )
    original = result["samples"]
    samples = [
        sample
        for sample in original
        if (
            sample["M"],
            sample["head_dim"],
            sample["num_heads_q"],
            sample["num_heads_kv"],
        )
        in allowed
    ]
    result["excluded_sample_count"] = len(original) - len(samples)
    result["samples"] = samples
    result["shape_policy"] = "workload_rows_80pct_l2"
    result["model"] = args.model
    result["mode"] = args.mode
    launch_latency = device_dict[args.device].compute_module.launch_latency.rope
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["fit"] = fit_samples(samples, launch_latency)
    directory = ROOT / "temp"
    directory.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"compute_latency_factor.{args.device}.{stamp}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for sample in samples:
        print(
            f"M={sample['M']} N={sample['N']} Average Latency: {sample['latency_seconds'] * 1000:.3f} ms"
        )
    print(json.dumps(result["fit"], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
