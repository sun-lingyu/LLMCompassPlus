import argparse
import json
import os
from math import inf
from typing import Optional

from software_model.utils import DataType, data_type_dict
from test.flashattn.test_perf import flash_attn_min_latency_remote
from test.flashattn.utils import get_model_shape, get_output_dtype
from test.utils import run_power_monitor

file_dir = os.path.dirname(os.path.abspath(__file__))


def get_flash_attn_full_cmd(
    seq_len_q: int,
    seq_len_kv: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_causal: bool,
    is_prefill: bool,
    precision: str,
    output_dtype: DataType,
    device: str,
    port: int,
    host: str,
    user: str,
    total_duration: int,
    python_path: str,
):
    num_splits_list = [1] if is_prefill else [1, 2, 4]
    best_runtime = inf
    best_num_splits = None
    best_pack_gqa = None
    for num_splits in num_splits_list:
        for pack_gqa in [True, False]:
            best_runtime_temp, _ = flash_attn_min_latency_remote(
                seq_len_q,
                seq_len_kv,
                num_heads_q,
                num_heads_kv,
                head_dim,
                num_splits,
                is_causal,
                is_prefill,
                pack_gqa,
                precision,
                output_dtype,
                f"{file_dir}/temp/flashattn_perf_log.{args.device}.json",
                port,
            )
            if best_runtime_temp < best_runtime:
                best_runtime = best_runtime_temp
                best_num_splits = num_splits
                best_pack_gqa = pack_gqa
    _, (fa2_latency, fa3_latency) = flash_attn_min_latency_remote(
        seq_len_q,
        seq_len_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        best_num_splits,
        is_causal,
        is_prefill,
        best_pack_gqa,
        precision,
        output_dtype,
        f"{file_dir}/temp/flashattn_perf_log.{args.device}.json",
        port,
        ignore_cache=True,  # important: get fa2 and fa3 latency respectively
    )

    mode = "prefill" if is_prefill else "decode"
    python_cmd = [
        python_path,
        f"benchmark_flash_attn_{mode}.py",
        "-b",
        "1",
        "-s",
        f"{seq_len_kv}",
        "-hq",
        f"{num_heads_q}",
        "-hkv",
        f"{num_heads_kv}",
        "-d",
        f"{head_dim}",
        "--num_splits",
        f"{num_splits}",
    ]
    if is_causal:
        python_cmd.append("--causal")
    if pack_gqa:
        python_cmd.append("--pack_gqa")
    if is_prefill:
        assert seq_len_q == seq_len_kv
    else:
        assert not is_causal
        python_cmd.append(f"-q {seq_len_q}")
    if fa2_latency < fa3_latency:
        python_cmd.append("--fa2_only")
    else:
        python_cmd.append("--fa3_only")
    python_cmd.append("--duration")
    python_cmd.append(f"{total_duration * 1000}")

    return " ".join(python_cmd)


def measure_power_remote(
    seq_len_q: int,
    seq_len_kv: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_causal: bool,
    is_prefill: bool,
    precision: str,
    device: str,
    power_log: str,
    total_duration: int = 5,
    valid_duration: int = 1,
    host: str = "202.120.39.3",
    user: Optional[str] = "sly",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
    ignore_cache: bool = False,
) -> float:
    print(precision)
    output_dtype = get_output_dtype(data_type_dict[precision], True)
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
                                    r["seq_len_q"],
                                    r["seq_len_kv"],
                                    r["num_heads_q"],
                                    r["num_heads_kv"],
                                    r["head_dim"],
                                    r["is_causal"],
                                    r["precision"],
                                    r["output_dtype"],
                                )
                                func._cache_dict[key] = (r["power_GPU"], r["power_MEM"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Load power_log failed: {e}")
                func._all_records = []
                func._cache_dict = {}
    search_key = (
        seq_len_q,
        seq_len_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        is_causal,
        precision,
        output_dtype.name,
    )
    if not ignore_cache and search_key in func._cache_dict:
        return func._cache_dict[search_key]

    port = 9129 if device == "Orin" else 9147

    full_cmd = get_flash_attn_full_cmd(
        seq_len_q,
        seq_len_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        is_causal,
        is_prefill,
        precision,
        output_dtype,
        device,
        port,
        host,
        user,
        total_duration,
        python_path,
    )

    avg_power_GPU, avg_power_MEM = run_power_monitor(
        full_cmd, total_duration, valid_duration, device, user, host, port
    )

    if not ignore_cache:
        new_record = {
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "num_heads_q": num_heads_q,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "is_causal": is_causal,
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

    head_dim, num_heads_q, num_heads_kv = get_model_shape(args.model)
    seq_len_kv_list = [512, 768, 1024, 1280, 1536]
    is_prefill = is_causal = args.mode == "prefill"
    if args.model == "InternVision":
        assert args.mode == "prefill"
        seq_len_kv_list = [576, 1024]  # 336x336/448x448 with patch 14x14
        is_causal = False
    assert num_heads_q % 4 == 0 and num_heads_kv % 4 == 0

    for seq_len_kv in seq_len_kv_list:
        if args.mode == "prefill":
            seq_len_q_list = [seq_len_kv]
        else:  # decode
            if args.device == "Orin":
                seq_len_q_list = [64, 128]
            elif args.device == "Thor":
                seq_len_q_list = [64, 128] if args.precision != "fp4" else [128]
            else:
                raise ValueError("Unsupported device")

        for seq_len_q in seq_len_q_list:
            test_problems = [
                (seq_len_q, seq_len_kv, num_heads_q, num_heads_kv),
                (seq_len_q, seq_len_kv, num_heads_q // 2, num_heads_kv // 2),
                (seq_len_q, seq_len_kv, num_heads_q // 4, num_heads_kv // 4),
            ]
            for problem in test_problems:
                p1, p2 = measure_power_remote(
                    *problem,
                    head_dim,
                    is_causal,
                    is_prefill,
                    args.precision,
                    args.device,
                    f"{file_dir}/temp/power_log.{args.device}.json",
                    total_duration=10,
                )
                print(
                    f"seq_len_q {seq_len_q}, seq_len_kv {seq_len_kv}, num_heads_q {num_heads_q}, num_heads_kv {num_heads_kv}, precision {args.precision} Power GPU {p1:.2f}W Power MEM {p2:.2f}W"
                )
