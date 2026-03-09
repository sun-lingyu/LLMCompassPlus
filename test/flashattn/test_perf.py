import argparse
import json
import os
import re
import shutil
from math import inf
from typing import Optional

import pandas as pd

from hardware_model.device import device_dict
from software_model.flashattn import FlashAttn
from software_model.flashattn_combine import FlashAttentionCombine
from software_model.utils import DataType, Tensor, data_type_dict
from test.flashattn.utils import get_model_shape, get_output_dtype
from test.utils import run_remote_command

file_dir = os.path.dirname(os.path.abspath(__file__))


def flash_attn_min_latency_remote(
    seq_len_q: int,
    seq_len_kv: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    num_splits: int,
    is_causal: bool,
    is_prefill: bool,
    pack_gqa: bool,
    precision: str,
    output_dtype: DataType,
    flash_attn_perf_log: str,
    port: int,
    host: str = "202.120.39.3",
    user: Optional[str] = "sly",
    work_dir: str = "/home/sly",
    python_path: str = "/home/sly/anaconda3/envs/llmcompass/bin/python",
    ignore_cache: bool = False,
):
    func = flash_attn_min_latency_remote
    if not hasattr(func, "_cache_dict"):
        func._all_records = []
        func._cache_dict = {}
        if os.path.exists(flash_attn_perf_log):
            try:
                with open(flash_attn_perf_log, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        func._all_records = data
                        for r in data:
                            key = (
                                r["seq_len_q"],
                                r["seq_len_kv"],
                                r["num_heads_q"],
                                r["num_heads_kv"],
                                r["head_dim"],
                                r["num_splits"],
                                r["is_causal"],
                                r["pack_gqa"],
                                r["precision"],
                                r["output_dtype"],
                            )
                            func._cache_dict[key] = r["min_runtime"]

            except (json.JSONDecodeError, KeyError):
                func._all_records = []
                func._cache_dict = {}
    search_key = (
        seq_len_q,
        seq_len_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        num_splits,
        is_causal,
        pack_gqa,
        precision,
        output_dtype.name,
    )
    if not ignore_cache and search_key in func._cache_dict:
        return func._cache_dict[search_key], (-1, -1)

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

    output = run_remote_command(user, host, port, python_cmd, work_dir=work_dir)

    num = r"(inf|[0-9]+(?:\.[0-9]+)?)"
    pat = re.compile(rf"(FA2|FA3|Average Latency):\s*{num}\s*ms", re.IGNORECASE)

    vals = {m.group(1).lower(): float(m.group(2)) for m in pat.finditer(output)}

    if "average latency" not in vals:
        raise RuntimeError(
            "No 'Average Latency: xxx ms' found in remote output.\n"
            f"Python Command: {' '.join(python_cmd)}\n"
            f"Output:\n{output}"
        )

    best_runtime = vals["average latency"]
    fa2_latency = vals.get("fa2", inf)
    fa3_latency = vals.get("fa3", inf)

    if not ignore_cache:
        new_record = {
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "num_heads_q": num_heads_q,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "num_splits": num_splits,
            "is_causal": is_causal,
            "pack_gqa": pack_gqa,
            "precision": precision,
            "output_dtype": output_dtype.name,
            "min_runtime": best_runtime,
        }

        func._all_records.append(new_record)
        func._cache_dict[search_key] = best_runtime
        with open(flash_attn_perf_log, "w") as f:
            json.dump(func._all_records, f, indent=4, ensure_ascii=False)

    return best_runtime, (fa2_latency, fa3_latency)


def test_and_save_latency(
    test_problems: list,
    file_name: str,
    precision: str,
    device: str,
    is_prefill: bool,
    is_causal: bool,
):
    if precision == "fp16":
        qkv_dtype = data_type_dict["fp16"]
        intermediate_dtype = data_type_dict["fp32"]
        assert device == "Orin", "fp16 precision is for Orin only"
    elif precision == "fp8":
        qkv_dtype = data_type_dict["fp8"]
        intermediate_dtype = data_type_dict["fp32"]
        assert device == "Thor", "fp8 precision is for Thor only"
    else:
        raise ValueError("Unsupported precision")
    output_dtype = get_output_dtype(data_type_dict[precision], True)

    latency_list = []
    for idx, (seq_len_q, seq_len_kv, num_heads_q, num_heads_kv) in enumerate(
        test_problems
    ):
        print(
            f"seq_len_q: {seq_len_q}, seq_len_kv: {seq_len_kv}, num_heads_q: {num_heads_q}, num_heads_kv: {num_heads_kv}"
        )
        latency = inf
        roofline_latency = inf
        measurement_latency = inf
        num_splits_list = [1] if is_prefill else [1, 2, 4]
        for num_splits in num_splits_list:
            model = FlashAttn(
                qkv_dtype,
                intermediate_dtype,
                output_dtype,
                is_prefill,
                is_causal,
                num_splits,
                device,
            )
            _ = model(
                Tensor([seq_len_q, num_heads_q, head_dim], dtype=qkv_dtype),
                Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
                Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
            )
            latency_this = 1000 * (
                model.compile_and_simulate(pcb)
                + pcb.compute_module.launch_latency.flashattn
            )
            roofline_latency_this = 1000 * model.roofline_model(pcb)
            if num_splits > 1:
                model1 = FlashAttentionCombine(intermediate_dtype, output_dtype)
                _ = model1(
                    Tensor(
                        [seq_len_q, num_heads_q * head_dim, num_splits],
                        dtype=intermediate_dtype,
                    )
                )
                latency_this += 1000 * (
                    model1.compile_and_simulate(pcb)
                    + pcb.compute_module.launch_latency.flashattn_combine
                )
                roofline_latency_this += 1000 * model1.roofline_model(pcb)
            latency = min(latency, latency_this)
            roofline_latency = min(roofline_latency, roofline_latency_this)

            port = 9129 if args.device == "Orin" else 9147
            measurement_latency_this = min(
                flash_attn_min_latency_remote(
                    seq_len_q,
                    seq_len_kv,
                    num_heads_q,
                    num_heads_kv,
                    head_dim,
                    num_splits,
                    is_causal,
                    is_prefill,
                    pack_gqa_val,
                    precision,
                    output_dtype,
                    f"{file_dir}/temp/flashattn_perf_log.{args.device}.json",
                    port,
                )[0]
                for pack_gqa_val in [False, True]
            )
            measurement_latency = min(measurement_latency, measurement_latency_this)
        print(
            f"latency {latency:.3f}, roofline_latency {roofline_latency:.3f}, measurement_latency {measurement_latency:.3f}"
        )
        print()
        latency_list.append((latency, roofline_latency, measurement_latency))

    df = pd.DataFrame(latency_list, columns=["Ours", "Roofline", "Measurement"])
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
    head_dim, num_heads_q, num_heads_kv = get_model_shape(args.model)
    seq_len_kv_list = [512, 768, 1024, 1280, 1536]
    is_prefill = is_causal = args.mode == "prefill"
    if args.model == "InternVision":
        assert args.mode == "prefill"
        seq_len_kv_list = [576, 1024]  # 336x336/448x448 with patch 14x14
        is_causal = False

    target_dir = f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(
        target_dir,
        exist_ok=True,
    )
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
            test_and_save_latency(
                test_problems,
                f"{target_dir}/flashattn.csv",
                args.precision,
                args.device,
                is_prefill,
                is_causal,
            )
