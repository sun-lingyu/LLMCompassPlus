import argparse
import base64
import json
import os
import subprocess
from typing import Optional

from software_model.utils import data_type_dict
from test.matmul.test_perf import cutlass_gemm_min_latency_remote
from test.matmul.utils import get_model_shape, get_output_dtype

file_dir = os.path.dirname(os.path.abspath(__file__))


def measure_power_remote(
    M: int,
    N: int,
    K: int,
    op_name: str,
    precision: str,
    device: str,
    total_duration: int = 5,
    valid_duration: int = 1,
    host: str = "202.120.39.3",
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly/marlin",  # for marlin
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",  # for marlin
    profiler_path: str = "/home/sly/cutlass/build/tools/profiler/cutlass_profiler",  # for cutlass
    power_log: str = f"{file_dir}/temp/power_log.json",
    ignore_cache: bool = False,
) -> float:
    existing_data = []
    if os.path.exists(power_log):
        with open(power_log, "r") as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    existing_data = data
                else:
                    assert False, "power_log.json format error"

    if not ignore_cache:
        for record in existing_data:
            if (
                record.get("M") == M
                and record.get("N") == N
                and record.get("K") == K
                and record.get("precision") == precision
            ):
                return record["power_VDD_GPU_SOC"], record["power_VDDQ_VDD2_1V8AO"]

    port = 9129 if device == "Orin" else 9147

    print(
        f"Measuring power for {total_duration}s and take {valid_duration}s in the middle..."
    )

    if precision == "int4":  # marlin
        full_cmd = (
            f"{python_path} {work_dir}/test_simple.py {M} {N} {K} "
            f"--duration={(total_duration) * 1000} "  # ms
        )
    else:  # CUTLASS
        output_dtype = get_output_dtype(data_type_dict[precision], op_name, True)
        _, best_op_name = cutlass_gemm_min_latency_remote(
            M, N, K, precision, output_dtype, port, host, user, profiler_path
        )
        full_cmd = (
            f"{profiler_path} --m={M} --n={N} --k={K} --kernels={best_op_name} "
            f"--profiling-duration={(total_duration) * 1000} "  # ms
            "--profiling-iterations=0 "
            "--verification-enabled=false "
            "--warmup-iterations=0 "
        )

    print(full_cmd)

    remote_script_path = os.path.join(os.path.dirname(file_dir), "power_monitor.py")
    if not os.path.exists(remote_script_path):
        raise FileNotFoundError(f"Cannot find remote script at {remote_script_path}")

    with open(remote_script_path, "r", encoding="utf-8") as f:
        script_body = f.read()

    variables_header = f"""
FULL_CMD = {repr(full_cmd)}
VALID_START_TIME = {total_duration / 2 - valid_duration / 2}
VALID_DURATION = {valid_duration}
DEVICE = "{device}"
"""
    remote_script_source = variables_header + "\n" + script_body

    b64_script = base64.b64encode(remote_script_source.encode("utf-8")).decode("utf-8")

    target = f"{user}@{host}" if user is not None else host
    ssh_cmd = [
        "ssh",
        "-p",
        str(port),
        target,
        f"echo {b64_script} | base64 -d | python3",
    ]

    proc = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    output = proc.stdout.strip()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Power measurement failed (code {proc.returncode}).\nOutput:\n{output}"
        )

    try:
        avg_power_VDD_GPU_SOC = float(output.splitlines()[-2])
        avg_power_VDDQ_VDD2_1V8AO = float(output.splitlines()[-1])
    except ValueError:
        raise RuntimeError(f"Could not parse power output. Received:\n{output}")

    if not ignore_cache:
        new_record = {
            "M": M,
            "N": N,
            "K": K,
            "precision": precision,
            "power_VDD_GPU_SOC": avg_power_VDD_GPU_SOC,
            "power_VDDQ_VDD2_1V8AO": avg_power_VDDQ_VDD2_1V8AO,
        }

        existing_data.append(new_record)

        with open(power_log, "w") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

    return avg_power_VDD_GPU_SOC, avg_power_VDDQ_VDD2_1V8AO


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
    args = parser.parse_args()

    K_shapes, N_shapes = get_model_shape(args.model)
    M = 1024 if args.mode == "prefill" else 64
    assert M % 4 == 0
    if args.precision == "fp4" and args.mode == "decode":
        M = 128  # at least 128

    for op_name in ["qkv_proj", "o_proj", "up_proj", "down_proj"]:
        if args.op_name != "all" and args.op_name != op_name:
            continue

        K, N = K_shapes[op_name], N_shapes[op_name]
        assert N % 4 == 0 and K % 4 == 0

        if args.mode == "prefill":
            # op_name Context Parallelism
            test_problems = [(M, N, K), (M // 2, N, K), (M // 4, N, K)]
            for problem in test_problems:
                p1, p2 = measure_power_remote(
                    *problem, op_name, args.precision, args.device
                )
                print(
                    f"M N K {problem}, precision {args.precision} Power VDD_GPU_SOC {p1:.2f}W Power VDDQ_VDD2_1V8AO {p2:.2f}W"
                )

        # test Tensor Parallelism
        if op_name == "qkv_proj" or op_name == "up_proj":
            test_problems = [(M, N, K), (M, N // 2, K), (M, N // 4, K)]
        else:  # o_proj or down_proj
            test_problems = [(M, N, K), (M, N, K // 2), (M, N, K // 4)]
        for problem in test_problems:
            p1, p2 = measure_power_remote(
                *problem, op_name, args.precision, args.device
            )
            print(
                f"M N K {problem}, precision {args.precision} Power VDD_GPU_SOC {p1:.2f}W Power VDDQ_VDD2_1V8AO {p2:.2f}W"
            )
