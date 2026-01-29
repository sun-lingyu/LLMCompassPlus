import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Optional, Tuple, Union

import pandas as pd

from hardware_model.device import device_dict
from software_model.matmul import Matmul
from software_model.utils import DataType, Tensor, data_type_dict
from test.matmul.utils import get_model_shape, get_output_dtype

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
        precision,
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
    output_dtype: DataType,
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
        with open(cutlass_perf_log, "r") as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    existing_data = data
                else:
                    assert False, "cutlass_perf_log.json format error"

    for record in existing_data:
        if (
            record.get("M") == M
            and record.get("N") == N
            and record.get("K") == K
            and record.get("precision") == precision
        ):
            return record["min_runtime"], record["best_operation"]

    if precision == "fp16":
        cutlass_cmd = [
            profiler_path,
            "--operation=Gemm",
            f"--m={M}",
            f"--n={N}",
            f"--k={K}",
            "--A=f16:row",
            "--B=f16:column",
            "--D=f16",
            "--accum=f32",
            "--beta=0",
        ]
    elif precision == "int8":
        cutlass_cmd = [
            profiler_path,
            "--operation=Gemm",
            f"--m={M}",
            f"--n={N}",
            f"--k={K}",
            "--A=s8:row",
            "--B=s8:column",
            "--D=s8",
            "--accum=s32",
            "--beta=0",
        ]
    elif precision == "fp8":
        assert output_dtype.name in ("fp8", "fp16")
        kernel_prefix = "cutlass3x_sm100_tensorop_gemm_f8_f8_f32_void_"
        if output_dtype.name == "fp8":
            kernel_prefix += "e4m3"
        else:
            kernel_prefix += "bf16"
        cutlass_cmd = [
            profiler_path,
            f"--m={M}",
            f"--n={N}",
            f"--k={K}",
            f"--kernels={kernel_prefix}*",
        ]
    elif precision == "fp4":
        assert output_dtype.name in ("fp4", "fp8", "fp16")
        kernel_prefix = (
            "cutlass3x_sm100_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_"  # nvfp4
        )
        if output_dtype.name == "fp4":
            kernel_prefix += "ue8m0xe2m1"
        elif output_dtype.name == "fp8":
            kernel_prefix += (
                "e5m2"  # Ideally this should be e4m3, but CUTLASS does not support.
            )
        else:
            kernel_prefix += "e5m2"  # This should be fp16/bf16 if supported
        cutlass_cmd = [
            profiler_path,
            f"--m={M}",
            f"--n={N}",
            f"--k={K}",
            f"--kernels={kernel_prefix}*",
        ]
    else:
        assert False, "Precision not supported"

    cutlass_cmd += [
        "--llc-capacity=524288"
    ]  # llc-capacity set to very large 512MB to avoid any L2 residency

    # cutlass_cmd += ["--enable-best-kernel-for-fixed-shape"] # Takes very long time but gives similar results

    # 1. Fast scanning, might be inaccurate
    remote_cmd_str = " ".join(shlex.quote(arg) for arg in cutlass_cmd)
    print(remote_cmd_str)
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
    best_runtime_temp, best_op_name = min(results, key=lambda x: x[0])

    # 2. Check best_op again
    cutlass_cmd = [
        profiler_path,
        f"--m={M}",
        f"--n={N}",
        f"--k={K}",
        f"--kernels={best_op_name}",
        "--llc-capacity=524288",
        "--profiling-duration=100",
        "--profiling-iterations=0",  # Run for 100ms
        "--warmup-iterations=100",
        "--enable-best-kernel-for-fixed-shape",
    ]
    remote_cmd_str = " ".join(shlex.quote(arg) for arg in cutlass_cmd)
    print(remote_cmd_str)
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
    best_runtime = None
    results = []
    for line in output.splitlines():
        line = line.strip()

        if "Runtime:" in line and current_op:
            match = re.search(r"Runtime:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                results.append(float(match.group(1)))
    best_runtime = min(results)
    if not best_runtime:
        raise RuntimeError(
            "No valid performance data found in profiler output.\n"
            f"Output snippet:\n{output[:1000]}"
        )

    new_record = {
        "M": M,
        "N": N,
        "K": K,
        "precision": precision,
        "min_runtime": best_runtime,
        "best_operation": best_op_name,
    }

    existing_data.append(new_record)

    with open(cutlass_perf_log, "w") as f:
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
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
    marlin_perf_log: str = f"{file_dir}/temp/marlin_perf_log.json",
) -> float:
    existing_data = []
    if os.path.exists(marlin_perf_log):
        with open(marlin_perf_log, "r") as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    existing_data = data
                else:
                    assert False, "marlin_perf_log.json format error"

    for record in existing_data:
        if record.get("M") == M and record.get("N") == N and record.get("K") == K:
            return record["min_runtime"]

    python_cmd = [python_path, "test_simple.py", str(M), str(N), str(K)]

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
    best_runtime = float(match.group(1))
    new_record = {
        "M": M,
        "N": N,
        "K": K,
        "min_runtime": best_runtime,
    }

    existing_data.append(new_record)

    with open(marlin_perf_log, "w") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    return best_runtime


def test_and_save_latency(
    test_problems: list,
    file_name: str,
    precision: str,
    op_name: str,
    update_ours_only: bool,
    device: str,
):
    if update_ours_only:
        df = pd.read_csv(file_name)
        assert len(test_problems) == df.shape[0]

    if precision == "fp16":
        activation_dtype = data_type_dict["fp16"]
        weight_dtype = data_type_dict["fp16"]
        intermediate_dtype = data_type_dict["fp32"]
        assert device == "Orin", "fp16 precision is for Orin only"
    elif precision == "int8":
        activation_dtype = data_type_dict["int8"]
        weight_dtype = data_type_dict["int8"]
        intermediate_dtype = data_type_dict["int32"]
        assert device == "Orin", "int8 precision is for Orin only"
    elif precision == "int4":
        activation_dtype = data_type_dict["fp16"]
        weight_dtype = data_type_dict["int4"]
        intermediate_dtype = data_type_dict["fp32"]
        assert device == "Orin", "int4 precision is for Orin only"
    elif precision == "fp8":
        activation_dtype = data_type_dict["fp8"]
        weight_dtype = data_type_dict["fp8"]
        intermediate_dtype = data_type_dict["fp32"]
        assert device == "Thor", "fp8 precision is for Thor only"
    elif precision == "fp4":
        activation_dtype = data_type_dict["fp4"]
        weight_dtype = data_type_dict["fp4"]
        intermediate_dtype = data_type_dict["fp32"]
        assert device == "Thor", "fp4 precision is for Thor only"
    output_dtype = get_output_dtype(activation_dtype, op_name, True)

    latency_list = []
    for idx, (M, N, K) in enumerate(test_problems):
        print(f"problem: M {M} N {N} K {K}")
        model = Matmul(
            activation_dtype, weight_dtype, intermediate_dtype, output_dtype, device
        )
        _ = model(
            Tensor([M, K], activation_dtype),
            Tensor([K, N], weight_dtype),
        )
        latency = 1000 * (
            model.compile_and_simulate(pcb) + pcb.compute_module.launch_latency.matmul
        )
        if update_ours_only:
            baseline_latency = (
                float(df["Baseline"].iloc[idx]) if precision != "int4" else -1
            )
            roofline_latency = float(df["Roofline"].iloc[idx])
            cutlass_latency = float(df["CUTLASS"].iloc[idx])
        else:
            baseline_latency = (
                get_baseline_latency(M, N, K, args.precision)
                if precision in ("int8", "fp16")
                else -1
            )
            roofline_latency = 1000 * model.roofline_model(pcb)
            port = 9129 if args.device == "Orin" else 9147
            cutlass_latency = (
                cutlass_gemm_min_latency_remote(M, N, K, precision, output_dtype, port)[
                    0
                ]
                if precision != "int4"
                else marlin_gemm_latency_remote(M, N, K)
            )
        print(
            f"latency {latency:.3f}, baseline_latency {baseline_latency:.3f}, roofline_latency {roofline_latency:.3f}, cutlass_latency {cutlass_latency:.3f}"
        )
        print()
        latency_list.append(
            (latency, baseline_latency, roofline_latency, cutlass_latency)
        )

    df = pd.DataFrame(latency_list, columns=["Ours", "Baseline", "Roofline", "CUTLASS"])
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"])
    parser.add_argument(
        "--op_name",
        type=str,
        choices=["qkv_proj", "o_proj", "up_proj", "down_proj", "all"],
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"],
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    parser.add_argument("--update_ours_only", action="store_true")
    args = parser.parse_args()

    pcb = device_dict[args.device]
    K_shapes, N_shapes = get_model_shape(args.model)
    M = 1024 if args.mode == "prefill" else 64
    assert M % 4 == 0
    if args.precision == "fp4" and args.mode == "decode":
        M = 128  # at least 128

    os.makedirs(
        f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}",
        exist_ok=True,
    )
    for op_name in ["qkv_proj", "o_proj", "up_proj", "down_proj"]:
        if args.op_name != "all" and args.op_name != op_name:
            continue

        K, N = K_shapes[op_name], N_shapes[op_name]
        assert N % 4 == 0 and K % 4 == 0

        if args.mode == "prefill":
            # test Context Parallelism
            test_problems = [(M, N, K), (M // 2, N, K), (M // 4, N, K)]
            test_and_save_latency(
                test_problems,
                f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{op_name}_CP.csv",
                args.precision,
                op_name,
                args.update_ours_only,
                args.device,
            )

        # test Tensor Parallelism
        if op_name == "qkv_proj" or op_name == "up_proj":
            test_problems = [(M, N, K), (M, N // 2, K), (M, N // 4, K)]
        else:  # o_proj or down_proj
            test_problems = [(M, N, K), (M, N, K // 2), (M, N, K // 4)]
        test_and_save_latency(
            test_problems,
            f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{op_name}_TP.csv",
            args.precision,
            op_name,
            args.update_ours_only,
            args.device,
        )
