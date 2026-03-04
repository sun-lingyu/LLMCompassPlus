import argparse
import base64
import json
import os
import re
from typing import Optional

from software_model.utils import data_type_dict
from test.matmul.test_perf import cutlass_gemm_min_latency_remote
from test.matmul.utils import get_model_shape, get_output_dtype
from test.utils import run_remote_command

file_dir = os.path.dirname(os.path.abspath(__file__))


def get_cutlass_full_cmd(
    M,
    N,
    K,
    precision,
    output_dtype,
    device,
    port,
    host,
    user,
    profiler_path,
    total_duration,
):
    _, best_op_name = cutlass_gemm_min_latency_remote(
        M,
        N,
        K,
        precision,
        output_dtype,
        f"{file_dir}/temp/cutlass_perf_log.{device}.json",
        port,
        host,
        user,
        profiler_path,
    )
    test_cmd = [
        f"{profiler_path}",
        f"--m={M}",
        f"--n={N}",
        f"--k={K}",
        f"--kernels={best_op_name}",
        "--llc-capacity=524288",
        "--profiling-duration=1000",
        "--profiling-iterations=0",  # Run for 1000ms
        "--warmup-iterations=1000",  # extend warm up iterations
        "--enable-best-kernel-for-fixed-shape",  # explore all configurations
    ]
    output = run_remote_command(user, host, port, test_cmd)
    match = re.search(r"Arguments:\s*(.*?)(?=\n\s+[A-Z])", output, re.DOTALL)
    args_str = match.group(1) if match else ""
    raw_args_list = args_str.replace("\\", " ").split()
    exclude = {
        "--runtime_input_datatype_a=invalid",
        "--runtime_input_datatype_b=invalid",
        "--op_class=invalid",
    }
    final_args_list = [arg for arg in raw_args_list if arg not in exclude]
    extra_args_list = [
        "--llc-capacity=524288",
        f"--profiling-duration={(total_duration) * 1000}",  # ms
        "--profiling-iterations=0",
        "--verification-enabled=false",
        "--warmup-iterations=0",
    ]
    return " ".join(
        [f"{profiler_path}", f"--kernels={best_op_name}"]
        + final_args_list
        + extra_args_list
    )


def measure_power_remote(
    M: int,
    N: int,
    K: int,
    op_name: str,
    precision: str,
    device: str,
    power_log: str,
    total_duration: int = 5,
    valid_duration: int = 1,
    host: str = "202.120.39.3",
    user: Optional[str] = "sly",
    marlin_path: str = "/home/sly/marlin",  # for marlin
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",  # for marlin
    profiler_path: str = "/home/sly/cutlass/build/tools/profiler/cutlass_profiler",  # for cutlass
    ignore_cache: bool = False,
) -> float:
    output_dtype = get_output_dtype(data_type_dict[precision], op_name, True)
    func = measure_power_remote
    if not hasattr(func, "_cache_dict"):
        func._all_records = []
        func._cache_dict = {}
        if os.path.exists(power_log):
            try:
                with open(power_log, "r") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
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
                                func._cache_dict[key] = (r["power_GPU"], r["power_MEM"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Load power_log failed: {e}")
                func._all_records = []
                func._cache_dict = {}
    search_key = (M, N, K, precision, output_dtype.name)
    if not ignore_cache and search_key in func._cache_dict:
        return func._cache_dict[search_key]

    port = 9129 if device == "Orin" else 9147

    if precision == "int4":  # marlin
        cmd_parts = [
            python_path,
            f"{marlin_path}/test_simple.py",
            str(M),
            str(N),
            str(K),
            f"--duration={total_duration * 1000}",
        ]
        full_cmd = " ".join(cmd_parts)
    else:  # CUTLASS
        full_cmd = get_cutlass_full_cmd(
            M,
            N,
            K,
            precision,
            output_dtype,
            device,
            port,
            host,
            user,
            profiler_path,
            total_duration,
        )

    print(
        f"Measuring power for {total_duration}s and take {valid_duration}s in the middle..."
    )
    print(full_cmd)

    power_monitor_path = os.path.abspath(
        os.path.join(file_dir, "..", "power_monitor.py")
    )
    if not os.path.exists(power_monitor_path):
        raise FileNotFoundError(f"Cannot find power_monitor at {power_monitor_path}")

    with open(power_monitor_path, "r", encoding="utf-8") as f:
        script_body = f.read()

    variables_header = f"""
FULL_CMD = {repr(full_cmd)}
VALID_START_TIME = {total_duration / 2 - valid_duration / 2}
VALID_DURATION = {valid_duration}
DEVICE = "{device}"
"""
    remote_script_source = variables_header + "\n" + script_body
    b64_script = base64.b64encode(remote_script_source.encode("utf-8")).decode("utf-8")
    shell_pipeline = f"echo {b64_script} | base64 -d | python3"
    remote_cmd = ["bash", "-c", shell_pipeline]
    output = run_remote_command(user, host, port, remote_cmd).strip()

    try:
        avg_power_GPU = float(output.splitlines()[-2])
        avg_power_MEM = float(output.splitlines()[-1])
    except ValueError:
        raise RuntimeError(f"Could not parse power output. Received:\n{output}")

    if not ignore_cache:
        new_record = {
            "M": M,
            "N": N,
            "K": K,
            "precision": precision,
            "output_dtype": output_dtype.name,
            "power_GPU": avg_power_GPU,
            "power_MEM": avg_power_MEM,
        }

        func._all_records.append(new_record)
        func._cache_dict[search_key] = (avg_power_GPU, avg_power_MEM)
        with open(power_log, "w") as f:
            json.dump(func._all_records, f, indent=4, ensure_ascii=False)

    return avg_power_GPU, avg_power_MEM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prefill", "decode"],
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"],
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()

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

    for M in M_list:
        assert M % 4 == 0
        for op_name in ["qkv_proj", "o_proj", "up_proj", "down_proj"]:
            K, N = K_shapes[op_name], N_shapes[op_name]
            assert N % 4 == 0 and K % 4 == 0

            if args.mode == "prefill":  # test Context Parallelism for prefill only
                test_problems = [(M, N, K), (M // 2, N, K), (M // 4, N, K)]
                for problem in test_problems:
                    p1, p2 = measure_power_remote(
                        *problem,
                        op_name,
                        args.precision,
                        args.device,
                        f"{file_dir}/temp/power_log.{args.device}.json",
                    )
                    print(
                        f"M N K {problem}, precision {args.precision} Power GPU {p1:.2f}W Power MEM {p2:.2f}W"
                    )

            # test Tensor Parallelism
            if op_name == "qkv_proj" or op_name == "up_proj":
                test_problems = [(M, N, K), (M, N // 2, K), (M, N // 4, K)]
            else:  # o_proj or down_proj
                test_problems = [(M, N, K), (M, N, K // 2), (M, N, K // 4)]
            for problem in test_problems:
                p1, p2 = measure_power_remote(
                    *problem,
                    op_name,
                    args.precision,
                    args.device,
                    f"{file_dir}/temp/power_log.{args.device}.json",
                )
                print(
                    f"M N K {problem}, precision {args.precision} Power GPU {p1:.2f}W Power MEM {p2:.2f}W"
                )
