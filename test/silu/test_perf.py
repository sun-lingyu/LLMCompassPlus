"""Compare the SiLU model with FlashInfer FP16 SiLU/Mul DRAM measurements.

From the repository root:
    python -m test.silu.test_perf --device Thor --model Qwen3_4B --mode prefill
    python -m test.silu.test_perf --device Orin --shape 64 3072
N denotes output width: input is [M, 2*N], output is [M, N].
Writes timestamped raw JSON and comparison CSV, without changing hardware config.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from hardware_model.device import device_dict
from software_model.silu import SiLU
from software_model.utils import Tensor, data_type_dict
from test.utils import run_remote_command, test_model_dict

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "target_scripts" / "benchmark_fused_silu.py"


def make_shapes(model, mode):
    config = test_model_dict[model]
    if config["hidden_act"] != "silu":
        raise ValueError(f"{model} does not use SiLU")
    width = config["intermediate_size"]
    rows = [512, 768, 1024, 1280, 1536] if mode == "prefill" else [64, 128]
    return [(m, width // degree) for m in rows for degree in (1, 2, 4)]


def silu_latency_remote(device, shapes):
    source = TARGET.read_text()
    output = run_remote_command(
        "sly",
        "202.120.39.3",
        9129 if device == "Orin" else 9147,
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
    samples = [
        json.loads(line[7:])
        for line in output.splitlines()
        if line.startswith("RESULT ")
    ]
    if [(s["M"], s["N"]) for s in samples] != shapes:
        raise ValueError("Incomplete or unexpected measurement results")
    return samples, hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["Orin", "Thor"])
    shape_mode = parser.add_mutually_exclusive_group()
    shape_mode.add_argument(
        "--model",
        choices=[k for k, v in test_model_dict.items() if v["hidden_act"] == "silu"],
    )
    shape_mode.add_argument("--shape", type=int, nargs=2, metavar=("M", "N"))
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--precision", choices=["fp16"], default="fp16")
    args = parser.parse_args()
    model_name = args.model or "Qwen3_4B"
    shapes = [tuple(args.shape)] if args.shape else make_shapes(model_name, args.mode)
    if any(m <= 0 or n <= 0 or n % 8 for m, n in shapes):
        parser.error("M must be positive; output N must be a positive multiple of 8")
    print(f"{args.device}: measuring {len(shapes)} FP16 SiLU/Mul shapes", flush=True)
    samples, source_hash = silu_latency_remote(args.device, shapes)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = ROOT / "temp"
    raw_dir.mkdir(exist_ok=True)
    raw_path = raw_dir / f"silu_perf_log.{args.device}.{stamp}.json"
    raw_path.write_text(
        json.dumps(
            {
                "device": args.device,
                "operator": "silu",
                "precision": "fp16",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "script_sha256": source_hash,
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )

    pcb = device_dict[args.device]
    dtype = data_type_dict["fp16"]
    rows = []
    for sample in samples:
        m, n = sample["M"], sample["N"]
        op = SiLU(dtype, dtype, fused_mul=True)
        out = op(Tensor([m, 2 * n], dtype))
        assert out.shape == [m, n]
        launch = pcb.compute_module.launch_latency.silu
        simulated_ms = (op.compile_and_simulate(pcb) + launch) * 1000
        row = {
            "M": m,
            "N": n,
            "Ours": simulated_ms,
            "Roofline": op.roofline_model(pcb) * 1000,
            "Measurement": sample["latency_ms"],
            "memory_latency_ms": op.memory_latency * 1000,
            "compute_latency_ms": op.compute_latency * 1000,
            "launch_latency_ms": launch * 1000,
        }
        rows.append(row)
        print(
            f"M={m} N={n}: Ours {simulated_ms:.3f} ms, Measurement {sample['latency_ms']:.3f} ms"
        )
    directory = (
        ROOT
        / "results_perf"
        / ("custom" if args.shape else model_name)
        / args.device
        / "fp16"
        / args.mode
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"silu.{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw measurements: {raw_path}")
    print(f"Comparison CSV (all times in ms): {path}")


if __name__ == "__main__":
    main()
