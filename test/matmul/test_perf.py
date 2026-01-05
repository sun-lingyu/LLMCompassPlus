from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from test.matmul.utils import get_model_shape

import os
import sys
import subprocess
import re
import shlex
import argparse
import json
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict, Any

file_dir = os.path.dirname(os.path.abspath(__file__))

def get_baseline_latency(
    M: int,
    N: int,
    K: int,
    precision: str,
    repo_root: Union[str, os.PathLike] = "~/LLM/LLMCompass",
) -> float:
    repo_root = os.path.expanduser(str(repo_root))
    cmd = [
        sys.executable,
        "-m",
        "test.matmul.test_matmul_simple",
        str(M),
        str(N),
        str(K),
        precision
    ]

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"baseline script exited with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"CWD: {repo_root}\n"
            f"Output:\n{output}"
        )

    m = re.search(r"Latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", output)
    if not m:
        raise RuntimeError(
            "Cannot find 'Latency: ... ms' in script output.\n"
            f"Command: {' '.join(cmd)}\n"
            f"CWD: {repo_root}\n"
            f"Output:\n{output}"
        )
    return float(m.group(1))

def cutlass_gemm_min_latency_remote(
    M: int,
    N: int,
    K: int,
    precision: str,
    port: int,
    host: str = "202.120.39.3",
    user: Optional[str] = "sly",
    profiler_path: str = "/home/sly/cutlass/build/tools/profiler/cutlass_profiler",
    cutlass_perf_log: str = f"{file_dir}/temp/cutlass_perf_log.json",
) -> Tuple[float, str, Optional[int]]:
    """
    Returns:
        (min_runtime, best_operation)
    """
    existing_data = []
    if os.path.exists(cutlass_perf_log):
        with open(cutlass_perf_log, 'r') as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    existing_data = data
                else:
                    assert False, "cutlass_perf_log.json format error"
    
    for record in existing_data:
        if (record.get("M") == M and 
            record.get("N") == N and 
            record.get("K") == K and 
            record.get("precision") == precision):
            return record['min_runtime'], record['best_operation']

    if precision == "fp16":
        cutlass_cmd = [
            profiler_path, "--operation=Gemm",
            f"--m={M}", f"--n={N}", f"--k={K}",
            "--A=f16:row", "--B=f16:column", "--D=f16",
            "--accum=f32", "--beta=0"
        ]
    elif precision == "int8":
        cutlass_cmd = [
            profiler_path, "--operation=Gemm",
            f"--m={M}", f"--n={N}", f"--k={K}",
            "--A=s8:row", "--B=s8:column", "--D=s8",
            "--accum=s32", "--beta=0"
        ]
    else:
        assert False, "Precision not supported"
    
    remote_cmd_str = " ".join(shlex.quote(arg) for arg in cutlass_cmd)
    target = f"{user}@{host}" if user is not None else host
    ssh_cmd = ["ssh", "-p", str(port), target, remote_cmd_str]

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
            f"ssh/cutlass_profiler exited with code {proc.returncode}\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )

    results = []
    current_op = None
    
    for line in output.splitlines():
        line = line.strip()
        
        if line.startswith("Operation:"):
            current_op = line.split(":", 1)[1].strip()
        
        if "Runtime:" in line and current_op:
            match = re.search(r"Runtime:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                runtime_ms = float(match.group(1))
                results.append((runtime_ms, current_op))

    if not results:
        raise RuntimeError(
            "No valid performance data found in profiler output.\n"
            f"Output snippet:\n{output[:1000]}"
        )

    best_runtime, best_op_name = min(results, key=lambda x: x[0])

    new_record = {
        "M": M, "N": N, "K": K, "precision": precision,
        "min_runtime": best_runtime, 
        "best_operation": best_op_name
    }

    existing_data.append(new_record)

    with open(cutlass_perf_log, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    return best_runtime, best_op_name

def marlin_gemm_latency_remote(
    M: int,
    N: int,
    K: int,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly/marlin",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python"
) -> float:
    python_cmd = [
        python_path,
        "test_simple.py",
        str(M),
        str(N),
        str(K)
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
            f"ssh/marlin exited with code {proc.returncode}\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )

    pattern = re.compile(r"average latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    match = pattern.search(output)
    
    if not match:
        raise RuntimeError(
            "No 'average latency: xxx ms' found in remote output.\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )

    return float(match.group(1))

def test_and_save_latency(
    test_problems: list,
    file_name:str,
    precision: str,
    update_ours_only: bool,
    port: int,
    ):

    if update_ours_only:
        df = pd.read_csv(file_name)
        assert(len(test_problems) == df.shape[0])
    
    if precision == "fp16":
        activation_data_type=data_type_dict["fp16"]
        weight_data_type=data_type_dict["fp16"]
        intermediate_data_type=data_type_dict["fp32"]
    elif precision == "int8":
        activation_data_type=data_type_dict["int8"]
        weight_data_type=data_type_dict["int8"]
        intermediate_data_type=data_type_dict["int32"]
    elif precision == "int4":
        activation_data_type=data_type_dict["fp16"]
        weight_data_type=data_type_dict["int4"]
        intermediate_data_type=data_type_dict["fp32"]
        assert port == 9129, "marlin int4 precision is for Orin only"

    latency_list = []
    for (idx, (M, N, K)) in enumerate(test_problems):
        print(f"problem: M {M} N {N} K {K}")
        model = Matmul(activation_data_type=activation_data_type, weight_data_type=weight_data_type, intermediate_data_type=intermediate_data_type)
        _ = model(
            Tensor([M, K], activation_data_type),
            Tensor([K, N], weight_data_type),
        )
        latency =  1000 * (model.compile_and_simulate(pcb, compile_mode="heuristic-GPU") + 2773 / pcb.compute_module.clock_freq)
        if update_ours_only:
            baseline_latency = float(df["Baseline"].iloc[idx]) if precision != "int4" else -1
            roofline_latency = float(df["Roofline"].iloc[idx])
            cutlass_latency = float(df["CUTLASS"].iloc[idx])
        else:
            baseline_latency =  get_baseline_latency(M, N, K, args.precision) if precision != "int4" else -1
            roofline_latency = 1000 * (model.roofline_model(pcb) + 2773 / pcb.compute_module.clock_freq)
            cutlass_latency = cutlass_gemm_min_latency_remote(M, N, K, precision, port)[0] if precision != "int4" else marlin_gemm_latency_remote(M, N, K)
        print(f"latency {latency:.3f}, baseline_latency {baseline_latency:.3f}, roofline_latency {roofline_latency:.3f}, cutlass_latency {cutlass_latency:.3f}")
        print()
        latency_list.append((latency, baseline_latency, roofline_latency, cutlass_latency))

    df = pd.DataFrame(
        latency_list,
        columns=["Ours", "Baseline", "Roofline", "CUTLASS"]
    )
    df.to_csv(file_name, index=False)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["Orin", "Thor"],)
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"])
    parser.add_argument("--test", type=str, choices=["qkv_proj", "o_proj", "up_proj", "down_proj", "all"])
    parser.add_argument("--model", type=str, choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"])
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"])
    parser.add_argument("--update_ours_only", action='store_true')
    args = parser.parse_args()
    
    pcb = device_dict[args.device]
    port = 9129 if args.device == "Orin" else 9147 # 9147: Thor
    K_shapes, N_shapes = get_model_shape(args.model)
    M = 1024 if args.mode == "prefill" else 64
    assert(M % 4 == 0)

    os.makedirs(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}", exist_ok=True)
    for test in ["qkv_proj", "o_proj", "up_proj", "down_proj"]:
        if args.test != "all" and args.test != test:
            continue

        K, N = K_shapes[test], N_shapes[test]
        assert(N % 4 == 0 and K % 4 == 0)

        if args.mode == "prefill":
            # test Context Parallelism
            test_problems = [(M, N, K), (M // 2, N, K), (M // 4, N, K)]
            test_and_save_latency(test_problems, f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{test}_CP.csv", args.precision, args.update_ours_only, port)

        # test Tensor Parallelism
        if test == "qkv_proj" or test == "up_proj":
            test_problems = [(M, N, K), (M, N // 2, K), (M, N // 4, K)]
        else: # o_proj or down_proj
            test_problems = [(M, N, K), (M, N, K // 2), (M, N, K // 4)]
        test_and_save_latency(test_problems, f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{test}_TP.csv", args.precision, args.update_ours_only, port)
    
        