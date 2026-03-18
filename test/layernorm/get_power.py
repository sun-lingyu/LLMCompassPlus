import argparse
import json
import os
from typing import Optional

from test.layernorm.utils import get_model_shape
from test.utils import run_power_monitor

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
                                )
                                func._cache_dict[key] = (r["power_GPU"], r["power_MEM"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Load power_log failed: {e}")
                func._all_records = []
                func._cache_dict = {}
    search_key = (M, N)
    if not ignore_cache and search_key in func._cache_dict:
        return func._cache_dict[search_key]

    port = 9129 if device == "Orin" else 9147

    python_cmd = [
        python_path,
        "benchmark_fused_rmsnorm.py",
        str(M),
        str(N),
        f"--duration={total_duration * 1000}",
    ]
    full_cmd = " ".join(python_cmd)

    avg_power_GPU, avg_power_MEM = run_power_monitor(
        full_cmd, total_duration, valid_duration, device, user, host, port
    )

    if not ignore_cache:
        new_record = {
            "M": M,
            "N": N,
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
    parser.add_argument("--precision", type=str, choices=["fp16"])
    args = parser.parse_args()

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
    N = get_model_shape(args.model)

    for M in M_list:
        assert M % 4 == 0
        test_problems = [(M, N), (M // 2, N), (M // 4, N)]
        for problem in test_problems:
            p1, p2 = measure_power_remote(
                *problem, args.device, f"{file_dir}/temp/power_log.{args.device}.json"
            )
            print(f"M {M} N {N}, Power GPU {p1:.2f}W Power MEM {p2:.2f}W")
