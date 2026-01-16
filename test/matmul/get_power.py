import os
import base64
import subprocess
import time
import argparse
import json
from typing import Optional
from test.matmul.test_perf import cutlass_gemm_min_latency_remote
from test.matmul.utils import get_model_shape
file_dir = os.path.dirname(os.path.abspath(__file__))

def measure_power_remote(
    M: int,
    N: int,
    K: int,
    precision: str,
    device: str,
    total_duration: int = 5,
    valid_duration: int = 1,
    host: str = "202.120.39.3",
    port: int = 9129,
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly/marlin", # for marlin
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python", # for marlin
    profiler_path: str = "/home/sly/cutlass/build/tools/profiler/cutlass_profiler", # for cutlass
    power_log: str = f"{file_dir}/temp/power_log.json",
    ignore_cache: bool = False
) -> float:
    existing_data = []
    if os.path.exists(power_log):
        with open(power_log, 'r') as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    existing_data = data
                else:
                    assert False, "power_log.json format error"

    if not ignore_cache:
        for record in existing_data:
            if (record.get("M") == M and 
                record.get("N") == N and 
                record.get("K") == K and 
                record.get("precision") == precision):
                return record['power_VDD_GPU_SOC'], record['power_VDDQ_VDD2_1V8AO']
        
    print(f"Measuring power for {total_duration}s and take {valid_duration}s in the middle...")

    if precision == "int4": # marlin
        full_cmd = (
            f"{python_path} {work_dir}/test_simple.py {M} {N} {K} "
            f"--duration={(total_duration) * 1000} " # ms
        )
    else: # CUTLASS
        _, best_op_name = cutlass_gemm_min_latency_remote(
            M, N, K, precision, host, output_dtype, port, user, profiler_path
        )
        full_cmd = (
            f"{profiler_path} --m={M} --n={N} --k={K} --kernels={best_op_name} "
            f"--profiling-duration={(total_duration) * 1000} " # ms
            "--profiling-iterations=0 "
            "--verification-enabled=false "
            "--warmup-iterations=0 "
        )

    print(full_cmd)
    
    volt_path_VDD_GPU_SOC = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input"
    curr_path_VDD_GPU_SOC = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"

    volt_path_VDDQ_VDD2_1V8AO = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2/in2_input"
    curr_path_VDDQ_VDD2_1V8AO = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2/curr2_input"

    remote_script_source = f"""
import time
import subprocess
import os
import signal
import sys

proc = subprocess.Popen('{full_cmd}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

samples = [] # (timestamp, power_watts)

try:
    while proc.poll() is None:        
        try:
            now = time.time()
            with open("{volt_path_VDD_GPU_SOC}", "r") as fv:
                mV_VDD_GPU_SOC = int(fv.read().strip())
            with open("{curr_path_VDD_GPU_SOC}", "r") as fc:
                mA_VDD_GPU_SOC = int(fc.read().strip())
            with open("{volt_path_VDDQ_VDD2_1V8AO}", "r") as fv:
                mV_VDDQ_VDD2_1V8AO = int(fv.read().strip())
            with open("{curr_path_VDDQ_VDD2_1V8AO}", "r") as fc:
                mA_VDDQ_VDD2_1V8AO = int(fc.read().strip())
            
            power_VDD_GPU_SOC = (mV_VDD_GPU_SOC / 1000.0) * (mA_VDD_GPU_SOC / 1000.0)
            power_VDDQ_VDD2_1V8AO  = (mV_VDDQ_VDD2_1V8AO / 1000.0) * (mA_VDDQ_VDD2_1V8AO / 1000.0)
            samples.append((now, power_VDD_GPU_SOC, power_VDDQ_VDD2_1V8AO))
        except Exception:
            pass # ignore temporary read errors
            
        time.sleep(0.05)

finally:
    if proc.poll() is None:
        proc.terminate()

end_time = samples[-1][0]
start_time = samples[0][0]
valid_start_time = (start_time + end_time) / 2 - {valid_duration} / 2
valid_samples = [(p1, p2) for (t, p1, p2) in samples if t >= valid_start_time and t <= valid_start_time + {valid_duration}]

if valid_samples:
    avg_power_VDD_GPU_SOC = sum([p[0] for p in valid_samples]) / len(valid_samples)
    avg_power_VDDQ_VDD2_1V8AO = sum([p[1] for p in valid_samples]) / len(valid_samples)
    print(avg_power_VDD_GPU_SOC)
    print(avg_power_VDDQ_VDD2_1V8AO)
else:
    print(0.0)
    print(0.0)
"""

    b64_script = base64.b64encode(remote_script_source.encode('utf-8')).decode('utf-8')
    
    target = f"{user}@{host}" if user is not None else host
    ssh_cmd = [
        "ssh",
        "-p", str(port),
        target,
        f"echo {b64_script} | base64 -d | python3"
    ]

    proc = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
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
            "M": M, "N": N, "K": K, "precision": precision,
            "power_VDD_GPU_SOC": avg_power_VDD_GPU_SOC, "power_VDDQ_VDD2_1V8AO": avg_power_VDDQ_VDD2_1V8AO
        }

        existing_data.append(new_record)

        with open(power_log, 'w') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    return avg_power_VDD_GPU_SOC, avg_power_VDDQ_VDD2_1V8AO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["Orin", "Thor"],)
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"],)
    parser.add_argument("--test", type=str, choices=["qkv_proj", "o_proj", "up_proj", "down_proj", "all"])
    parser.add_argument("--model", type=str, choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"])
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"])
    args = parser.parse_args()

    K_shapes, N_shapes = get_model_shape(args.model)
    M = 1024 if args.mode == "prefill" else 64
    assert(M % 4 == 0)

    for test in ["qkv_proj", "o_proj", "up_proj", "down_proj"]:
        if args.test != "all" and args.test != test:
            continue

        K, N = K_shapes[test], N_shapes[test]
        assert(N % 4 == 0 and K % 4 == 0)

        if args.mode == "prefill":
            # test Context Parallelism
            test_problems = [(M, N, K), (M // 2, N, K), (M // 4, N, K)]
            for problem in test_problems:
                p1, p2 = measure_power_remote(*problem, args.precision, args.device)
                print(f"M N K {problem}, precision {args.precision} Power VDD_GPU_SOC {p1:.2f}W Power VDDQ_VDD2_1V8AO {p2:.2f}W")

        # test Tensor Parallelism
        if test == "qkv_proj" or test == "up_proj":
            test_problems = [(M, N, K), (M, N // 2, K), (M, N // 4, K)]
        else: # o_proj or down_proj
            test_problems = [(M, N, K), (M, N, K // 2), (M, N, K // 4)]
        for problem in test_problems:
            p1, p2 = measure_power_remote(*problem, args.precision, args.device)
            print(f"M N K {problem}, precision {args.precision} Power VDD_GPU_SOC {p1:.2f}W Power VDDQ_VDD2_1V8AO {p2:.2f}W")
