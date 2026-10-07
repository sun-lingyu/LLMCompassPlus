import argparse
import json
import os
import re
import shutil
from typing import Optional

import pandas as pd

from hardware_model.device import device_dict
from software_model.rope import FusedQKNormRoPE
from software_model.utils import Tensor, data_type_dict
from test.rope.utils import get_model_shape, get_output_dtype
from test.utils import run_remote_command

file_dir = os.path.dirname(os.path.abspath(__file__))


def rope_latency_remote(
    M: int,
    head_dim: int,
    num_heads_q: int,
    num_heads_kv: int,
    device: str,
    rope_perf_log: str,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
) -> float:
    func = rope_latency_remote
    if not hasattr(func, "_cache_dict"):
        func._all_records = []
        func._cache_dict = {}
        if os.path.exists(rope_perf_log):
            try:
                with open(rope_perf_log, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        func._all_records = data
                        for r in data:
                            key = (
                                r["M"],
                                r["head_dim"],
                                r["num_heads_q"],
                                r["num_heads_kv"],
                            )
                            func._cache_dict[key] = r["min_runtime"]
            except (json.JSONDecodeError, KeyError):
                func._all_records = []
                func._cache_dict = {}
    search_key = (M, head_dim, num_heads_q, num_heads_kv)
    if search_key in func._cache_dict:
        return func._cache_dict[search_key]

    port = 9129 if device == "Orin" else 9147
    python_cmd = [
        "env",
        "PATH=/home/sly/anaconda3/envs/llmcompass/bin:$PATH",
        python_path,
        "benchmark_fused_qknorm_rope.py",
        str(M),
        str(head_dim),
        str(num_heads_q),
        str(num_heads_kv),
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
        "head_dim": head_dim,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "min_runtime": best_runtime,
    }

    func._all_records.append(new_record)
    func._cache_dict[search_key] = best_runtime
    with open(rope_perf_log, "w") as f:
        json.dump(func._all_records, f, indent=4, ensure_ascii=False)

    return best_runtime


def test_and_save_latency(
    test_problems: list,
    file_name: str,
    precision: str,
    device: str,
):
    input_dtype = data_type_dict[precision]
    output_dtype = get_output_dtype(input_dtype, True)
    latency_list = []
    for idx, (M, head_dim, num_heads_q, num_heads_kv) in enumerate(test_problems):
        print(
            f"problem: M {M} head_dim {head_dim} num_heads_q {num_heads_q} num_heads_kv {num_heads_kv}"
        )
        model = FusedQKNormRoPE(input_dtype, output_dtype)
        _ = model(
            Tensor([M, head_dim * (num_heads_q + num_heads_kv * 2)], input_dtype),
            head_dim,
            num_heads_q,
            num_heads_kv,
        )
        latency = 1000 * (
            model.compile_and_simulate(pcb) + pcb.compute_module.launch_latency.rope
        )
        roofline_latency = 1000 * model.roofline_model(pcb)
        port = 9129 if device == "Orin" else 9147
        measurement_latency = rope_latency_remote(
            M,
            head_dim,
            num_heads_q,
            num_heads_kv,
            device,
            f"{file_dir}/temp/rope_perf_log.{device}.json",
            port=port,
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
        head_dim, num_heads_q, num_heads_kv = get_model_shape(args.model)
        test_problems = [
            (M, head_dim, num_heads_q, num_heads_kv),
            (M, head_dim, num_heads_q // 2, num_heads_kv // 2),
            (M, head_dim, num_heads_q // 4, num_heads_kv // 4),
        ]
        test_and_save_latency(
            test_problems, f"{target_dir}/rope.csv", args.precision, args.device
        )
