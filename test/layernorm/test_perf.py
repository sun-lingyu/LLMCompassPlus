import os
import argparse
import json
import shlex
import subprocess
import re
import pandas as pd
from typing import Optional

from software_model.layernorm import FusedLayerNorm
from software_model.utils import data_type_dict, Tensor, DataType
from hardware_model.device import device_dict
from test.layernorm.utils import get_model_shape
file_dir = os.path.dirname(os.path.abspath(__file__))

def layernorm_latency_remote(
    M: int,
    N: int,
    device: str,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
    layernorm_perf_log: str = f"{file_dir}/temp/layernorm_perf_log.json",
) -> float:
    existing_data = []
    if os.path.exists(layernorm_perf_log):
        with open(layernorm_perf_log, 'r') as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    existing_data = data
                else:
                    assert False, "layernorm_perf_log.json format error"

    for record in existing_data:
        if (record.get("M") == M and 
            record.get("N") == N and
            record.get("device") == device):
            return record['min_runtime']
        
    port = 9129 if device == "Orin" else 9147
    python_cmd = [
        "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
        python_path,
        "benchmark_fused_rmsnorm.py",
        str(M),
        str(N)
    ]
    
    cmd_part = " ".join(shlex.quote(arg) for arg in python_cmd)
    remote_cmd_str = f"cd {work_dir} && {cmd_part}"

    target = f"{user}@{host}" if user is not None else host
    ssh_cmd = [
        "ssh",
        "-p",
        str(port),
        target,
        remote_cmd_str,
    ]

    proc = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    output = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh/python exited with code {proc.returncode}\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )

    pattern = re.compile(r"Average Latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    match = pattern.search(output)
    
    if not match:
        raise RuntimeError(
            "No 'Average Latency: xxx ms' found in remote output.\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )
    best_runtime = float(match.group(1))
    new_record = {
        "M": M, "N": N, "device": device,
        "min_runtime": best_runtime, 
    }

    existing_data.append(new_record)

    with open(layernorm_perf_log, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    return best_runtime

def test_and_save_latency(
    M: int,
    file_name:str,
    precision: str,
    update_ours_only: bool,
    device: str,
    ):
    test_problems = []
    for model in ["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"]:
        N = get_model_shape(model)
        test_problems.append((M, N))

    if update_ours_only:
        df = pd.read_csv(file_name)
        assert(len(model) == df.shape[0])

    assert precision == "fp16"
    
    latency_list = []
    for (idx, (M, N)) in enumerate(test_problems):
        print(f"problem: M {M} N {N}")
        model = FusedLayerNorm(data_type_dict["fp16"])
        _ = model(
            Tensor([M, N], data_type_dict["fp16"]),
            Tensor([M, N], data_type_dict["fp16"]),
        )
        latency =  1000 * max(model.compile_and_simulate(pcb), pcb.compute_module.launch_latency.layernorm)
        roofline_latency = 1000 * model.roofline_model(pcb)
        if update_ours_only:
            measured_latency = float(df["Measured"].iloc[idx])
        else:
            measured_latency = layernorm_latency_remote(M, N, args.device)
        print(f"latency {latency:.3f}, roofline_latency {roofline_latency:.3f}, measured_latency {measured_latency:.3f}")
        print()
        latency_list.append((latency, roofline_latency, measured_latency))

    df = pd.DataFrame(
        latency_list,
        columns=["Ours", "Roofline", "Measured"]
    )
    df.to_csv(file_name, index=False)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["Orin", "Thor"],)
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"])
    parser.add_argument("--precision", type=str, choices=["fp16"])
    parser.add_argument("--update_ours_only", action='store_true')
    args = parser.parse_args()
    
    pcb = device_dict[args.device]
    
    M = 1024 if args.mode == "prefill" else 64
    assert(M % 4 == 0)
    if args.precision == "fp4" and args.mode == "decode":
        M = 128 # at least 128

    os.makedirs(f"{file_dir}/results_perf/", exist_ok=True)
    test_and_save_latency(M, f"{file_dir}/results_perf/{args.device}_{args.mode}.csv", args.precision, args.update_ours_only, args.device)