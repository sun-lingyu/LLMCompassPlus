import argparse
import base64
import json
import os
from typing import Optional

from LLM.LLMCompassPlus.test.utils import run_remote_command

from test.layernorm.utils import get_model_shape

file_dir = os.path.dirname(os.path.abspath(__file__))


def measure_power_remote(
    M: int,
    N: int,
    device: str,
    power_log: str,
    total_duration: int = 10,
    valid_duration: int = 1,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",  # for marlin
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
                and record.get("device") == device
            ):
                return record["power_GPU"], record["power_MEM"]

    print(
        f"Measuring power for {total_duration}s and take {valid_duration}s in the middle..."
    )

    python_cmd = [
        "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
        python_path,
        "benchmark_fused_rmsnorm.py",
        str(M),
        str(N),
        f"--duration={total_duration * 1000}",
    ]

    full_cmd = " ".join(python_cmd)

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
            "device": device,
            "power_GPU": avg_power_GPU,
            "power_MEM": avg_power_MEM,
        }

        existing_data.append(new_record)

        with open(power_log, "w") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

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
    parser.add_argument("--precision", type=str, choices=["fp16"])
    args = parser.parse_args()

    M = 1024 if args.mode == "prefill" else 64
    assert M % 4 == 0

    test_problems = []
    for model in ["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"]:
        N = get_model_shape(model)
        test_problems.append((M, N))

    for problem in test_problems:
        p1, p2 = measure_power_remote(
            *problem, args.device, f"{file_dir}/temp/power_log.{args.device}.json"
        )
        print(f"M N {problem}, Power GPU {p1:.2f}W Power MEM {p2:.2f}W")
