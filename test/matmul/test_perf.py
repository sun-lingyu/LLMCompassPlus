import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Optional, Tuple, Union

import pandas as pd

from hardware_model.device import device_dict
from software_model.matmul import Matmul
from software_model.utils import DataType, Tensor, data_type_dict
from test.matmul.utils import get_model_shape, get_output_dtype
from test.utils import run_remote_command

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
    cutlass_perf_log: str,
    port: int,
    host: str = "202.120.39.3",
    user: Optional[str] = "sly",
    profiler_path: str = "/home/sly/cutlass/build/tools/profiler/cutlass_profiler",
) -> Tuple[float, str, Optional[int]]:
    """
    Returns:
        (min_runtime, best_operation)
    """
    func = cutlass_gemm_min_latency_remote
    if not hasattr(func, "_cache_dict"):
        func._all_records = []
        func._cache_dict = {}
        if os.path.exists(cutlass_perf_log):
            try:
                with open(cutlass_perf_log, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        func._all_records = data
                        for r in data:
                            key = (
                                r["M"],
                                r["N"],
                                r["K"],
                                r["precision"],
                                r["output_dtype"],
                            )
                            func._cache_dict[key] = (
                                r["min_runtime"],
                                r["best_operation"],
                            )
            except (json.JSONDecodeError, KeyError):
                func._all_records = []
                func._cache_dict = {}
    search_key = (M, N, K, precision, output_dtype.name)
    if search_key in func._cache_dict:
        return func._cache_dict[search_key]

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
            f"--kernels={kernel_prefix}*_tnt*",
        ]
    else:
        assert False, "Precision not supported"

    cutlass_cmd += [
        "--llc-capacity=524288"
    ]  # llc-capacity set to very large 512MB to avoid any L2 residency

    # cutlass_cmd += ["--enable-best-kernel-for-fixed-shape"] # Takes very long time

    # 1. Fast scanning, might be inaccurate
    output = run_remote_command(user, host, port, cutlass_cmd)
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
    top_results = sorted(results, key=lambda x: x[0])[:30]  # take top 30
    best_op_names = [r[1] for r in top_results]

    # 2. Check top best_op candidates again
    cutlass_cmd = [
        profiler_path,
        f"--m={M}",
        f"--n={N}",
        f"--k={K}",
        f"--kernels={','.join(best_op_names)}",
        "--llc-capacity=524288",
        "--profiling-duration=100",
        "--profiling-iterations=0",  # Run for 100ms
        "--warmup-iterations=100",
        "--enable-best-kernel-for-fixed-shape",  # explore all configurations of the top candidates
    ]
    output = run_remote_command(user, host, port, cutlass_cmd)
    best_runtime = None
    results = []
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
        "output_dtype": output_dtype.name,
        "min_runtime": best_runtime,
        "best_operation": best_op_name,
    }

    func._all_records.append(new_record)
    func._cache_dict[search_key] = (best_runtime, best_op_name)
    with open(cutlass_perf_log, "w") as f:
        json.dump(func._all_records, f, indent=4, ensure_ascii=False)

    return best_runtime, best_op_name


def marlin_gemm_latency_remote(
    M: int,
    N: int,
    K: int,
    output_dtype: DataType,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly/marlin",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
    marlin_perf_log: str = f"{file_dir}/temp/marlin_perf_log.json",
) -> float:
    func = marlin_gemm_latency_remote
    if not hasattr(func, "_cache_dict"):
        func._all_records = []
        func._cache_dict = {}
        if os.path.exists(marlin_perf_log):
            try:
                with open(marlin_perf_log, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        func._all_records = data
                        for r in data:
                            key = (
                                r["M"],
                                r["N"],
                                r["K"],
                                r["output_dtype"],
                            )
                            func._cache_dict[key] = r["min_runtime"]
            except (json.JSONDecodeError, KeyError):
                func._all_records = []
                func._cache_dict = {}
    search_key = (M, N, K, output_dtype.name)
    if search_key in func._cache_dict:
        return func._cache_dict[search_key]

    python_cmd = [python_path, "test_simple.py", str(M), str(N), str(K)]
    output = run_remote_command(user, host, port, python_cmd, work_dir=work_dir)

    pattern = re.compile(r"average latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    match = pattern.search(output)

    if not match:
        raise RuntimeError(
            "No 'average latency: xxx ms' found in remote output.\n"
            f"Python Command: {' '.join(python_cmd)}\n"
            f"Output:\n{output}"
        )
    best_runtime = float(match.group(1))
    new_record = {
        "M": M,
        "N": N,
        "K": K,
        "output_dtype": output_dtype.name,
        "min_runtime": best_runtime,
    }

    func._all_records.append(new_record)
    func._cache_dict[search_key] = best_runtime
    with open(marlin_perf_log, "w") as f:
        json.dump(func._all_records, f, indent=4, ensure_ascii=False)

    return best_runtime


def test_and_save_latency(
    test_problems: list,
    file_name: str,
    precision: str,
    op_name: str,
    device: str,
):
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
        baseline_latency = (
            get_baseline_latency(M, N, K, args.precision)
            if precision in ("int8", "fp16")
            else -1
        )
        roofline_latency = 1000 * model.roofline_model(pcb)
        port = 9129 if args.device == "Orin" else 9147
        measurement_latency = (
            cutlass_gemm_min_latency_remote(
                M,
                N,
                K,
                precision,
                output_dtype,
                f"{file_dir}/temp/cutlass_perf_log.{args.device}.json",
                port=port,
            )[0]
            if precision != "int4"
            else marlin_gemm_latency_remote(M, N, K, output_dtype, port=port)
        )
        print(
            f"latency {latency:.3f}, baseline_latency {baseline_latency:.3f}, roofline_latency {roofline_latency:.3f}, measurement_latency {measurement_latency:.3f}"
        )
        print()
        latency_list.append(
            (latency, baseline_latency, roofline_latency, measurement_latency)
        )

    df = pd.DataFrame(
        latency_list, columns=["Ours", "Baseline", "Roofline", "Measurement"]
    )
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
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()

    pcb = device_dict[args.device]
    K_shapes, N_shapes = get_model_shape(args.model)
    if args.mode == "prefill":
        M_list = [512, 768, 1024, 1280, 1536]
        if args.model == "InternVision":
            M_list = [576, 1024]  # 336x336/448x448 with patch 14x14
    elif args.mode == "decode":
        if args.device == "Orin":
            M_list = [64, 128]
        elif args.device == "Thor":
            M_list = [64, 128] if args.precision != "fp4" else [128]
        else:
            raise ValueError("Unsupported device")

    target_dir = f"{file_dir}/results_perf/{args.model}/{args.device}/{args.precision}/{args.mode}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(
        target_dir,
        exist_ok=True,
    )

    for M in M_list:
        assert M % 4 == 0
        for op_name in ["qkv_proj", "o_proj", "up_proj", "down_proj"]:
            K, N = K_shapes[op_name], N_shapes[op_name]
            assert N % 4 == 0 and K % 4 == 0

            if args.mode == "prefill":  # test Context Parallelism for prefill only
                test_problems = [(M, N, K), (M // 2, N, K), (M // 4, N, K)]
                test_and_save_latency(
                    test_problems,
                    f"{target_dir}/{op_name}_CP.csv",
                    args.precision,
                    op_name,
                    args.device,
                )

            # test Tensor Parallelism
            if op_name == "qkv_proj" or op_name == "up_proj":
                test_problems = [(M, N, K), (M, N // 2, K), (M, N // 4, K)]
            else:  # o_proj or down_proj
                test_problems = [(M, N, K), (M, N, K // 2), (M, N, K // 4)]
            test_and_save_latency(
                test_problems,
                f"{target_dir}/{op_name}_TP.csv",
                args.precision,
                op_name,
                args.device,
            )
