import argparse
import json
import os
import re
import shutil
from typing import Optional

import pandas as pd

from hardware_model.device import device_dict
from software_model.layernorm import FusedLayerNorm
from software_model.utils import Tensor, data_type_dict
from test.layernorm.utils import get_model_shape
from test.utils import run_remote_command

file_dir = os.path.dirname(os.path.abspath(__file__))


def layernorm_latency_remote(
    M: int,
    N: int,
    device: str,
    layernorm_perf_log: str,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
) -> float:
    func = layernorm_latency_remote
    if not hasattr(func, "_cache_dict"):
        func._all_records = []
        func._cache_dict = {}
        if os.path.exists(layernorm_perf_log):
            try:
                with open(layernorm_perf_log, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        func._all_records = data
                        for r in data:
                            key = (
                                r["M"],
                                r["N"],
                            )
                            func._cache_dict[key] = r["min_runtime"]
            except (json.JSONDecodeError, KeyError):
                func._all_records = []
                func._cache_dict = {}
    search_key = (M, N)
    if search_key in func._cache_dict:
        return func._cache_dict[search_key]

    port = 9129 if device == "Orin" else 9147
    python_cmd = [
        "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
        python_path,
        "benchmark_fused_rmsnorm.py",
        str(M),
        str(N),
    ]

    output = run_remote_command(user, host, port, python_cmd, work_dir=work_dir)

    pattern = re.compile(r"Average Latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    match = pattern.search(output)

    if not match:
        raise RuntimeError(
            "No 'Average Latency: xxx ms' found in remote output.\n"
            f"Python Command: {' '.join(python_cmd)}\n"
            f"Output:\n{output}"
        )
    best_runtime = float(match.group(1))
    new_record = {
        "M": M,
        "N": N,
        "min_runtime": best_runtime,
    }

    func._all_records.append(new_record)
    func._cache_dict[search_key] = best_runtime
    with open(layernorm_perf_log, "w") as f:
        json.dump(func._all_records, f, indent=4, ensure_ascii=False)

    return best_runtime


def test_and_save_latency(
    test_problems: list,
    file_name: str,
    precision: str,
    device: str,
):
    latency_list = []
    for idx, (M, N) in enumerate(test_problems):
        print(f"problem: M {M} N {N}")
        model = FusedLayerNorm(data_type_dict["fp16"])
        _ = model(
            Tensor([M, N], data_type_dict["fp16"]),
            Tensor([M, N], data_type_dict["fp16"]),
        )
        latency = 1000 * (
            model.compile_and_simulate(pcb)
            + pcb.compute_module.launch_latency.layernorm
        )
        roofline_latency = 1000 * model.roofline_model(pcb)
        port = 9129 if device == "Orin" else 9147
        measurement_latency = layernorm_latency_remote(
            M, N, device, f"{file_dir}/temp/layernorm_perf_log.{device}.json", port=port
        )
        print(
            f"latency {latency:.3f}, roofline_latency {roofline_latency:.3f}, measured_latency {measurement_latency:.3f}"
        )
        print()
        latency_list.append((latency, roofline_latency, measurement_latency))

    df = pd.DataFrame(latency_list, columns=["Ours", "Roofline", "Measurement"])
    if file_name:
        file_exists = os.path.exists(file_name)
        df.to_csv(file_name, mode="a", index=False, header=not file_exists)
    else:
        raise ValueError("file_name is required to save results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"])
    parser.add_argument(
        "--model",
        type=str,
        choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"],
    )
    parser.add_argument("--precision", type=str, choices=["fp16"])
    args = parser.parse_args()

    pcb = device_dict[args.device]

    if args.mode == "prefill":
        M_list = [512, 768, 1024, 1280, 1536]
        if args.model == "InternVision":
            M_list = [576, 1024]  # 336x336/448x448 with patch 14x14
    elif args.mode == "decode":
        M_list = [64, 128]

    target_dir = f"{file_dir}/results_perf/{args.model}/{args.device}/{args.precision}/{args.mode}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(
        target_dir,
        exist_ok=True,
    )

    for M in M_list:
        assert M % 4 == 0
        test_problems = []
        N = get_model_shape(args.model)
        test_problems.append((M, N))
        test_and_save_latency(
            test_problems, f"{target_dir}/layernorm.csv", args.precision, args.device
        )
