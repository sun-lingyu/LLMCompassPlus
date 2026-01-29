import argparse
import os
from typing import Any

import pandas as pd

from hardware_model.device import device_dict
from software_model.flashattention import FlashAttentionPrefill
from software_model.flashdecoding import FlashAttentionDecode
from software_model.utils import Tensor, data_type_dict
from test.matmul.utils import test_model_dict

file_dir = os.path.dirname(os.path.abspath(__file__))


def get_data_types(precision):
    if precision == "fp16":
        return data_type_dict["fp16"], data_type_dict["fp16"], data_type_dict["fp32"]
    elif precision == "int8":
        return data_type_dict["int8"], data_type_dict["int8"], data_type_dict["int32"]
    elif precision == "int4":
        return data_type_dict["fp16"], data_type_dict["int4"], data_type_dict["fp32"]
    return data_type_dict["fp16"], data_type_dict["fp16"], data_type_dict["fp32"]


def test_and_save_latency_prefill(
    test_params: list,
    file_name: str,
    precision: str,
    pcb: Any,
    args: argparse.Namespace,
):
    """
    Tests latency for FlashAttention Prefill phase and saves results to CSV.

    Args:
        test_params: List of tuples (batch_size, num_heads_q, num_heads_kv, head_dim, seq_len)
        file_name: Output CSV file path
        precision: Data precision (fp16, int8, etc.)
        pcb: Hardware configuration object
        args: Command line arguments
    """
    df = None
    if "all" not in args.update:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)

    qkv_data_type, output_data_type, intermediate_data_type = get_data_types(precision)

    latency_list = []
    for idx, (batch_size, num_heads_q, num_heads_kv, head_dim, seq_len) in enumerate(
        test_params
    ):
        print(
            f"problem: batch_size {batch_size} heads_q {num_heads_q} heads_kv {num_heads_kv} head_dim {head_dim} seq_len {seq_len}"
        )

        # Initialize Prefill Model
        model = FlashAttentionPrefill(
            qkv_data_type=qkv_data_type,
            output_data_type=output_data_type,
            intermediate_data_type=intermediate_data_type,
            pcb_module=pcb,
            is_causal=True,
        )

        # Configure model with input tensors
        _ = model(
            Tensor((batch_size, num_heads_q, seq_len, head_dim), qkv_data_type),
            Tensor((batch_size, num_heads_kv, seq_len, head_dim), qkv_data_type),
            Tensor((batch_size, num_heads_kv, seq_len, head_dim), qkv_data_type),
        )

        # 1. Analytical Model Simulation
        if "ours" in args.update or "all" in args.update or df is None:
            latency = 1000 * model.compile_and_simulate(compile_mode="exhaustive")
        else:
            latency = float(df["Ours"].iloc[idx])

        # 2. Roofline Model Estimation
        if "roofline" in args.update or "all" in args.update or df is None:
            roofline_latency = 1000 * model.roofline_model()
        else:
            roofline_latency = float(df["Roofline"].iloc[idx])

        # 3. Real GPU Measurement (Optional)
        if "gpu" in args.update or "all" in args.update or df is None:
            try:
                gpu_latency = 1000 * model.run_on_gpu()
            except Exception as e:
                print(f"Warning: GPU profiling failed: {e}")
                gpu_latency = -1
        else:
            gpu_latency = float(df["GPU"].iloc[idx])

        print(
            f"Ours: {latency:.3f} ms, Roofline: {roofline_latency:.3f} ms, GPU: {gpu_latency:.3f} ms"
        )
        latency_list.append(
            (
                batch_size,
                num_heads_q,
                num_heads_kv,
                head_dim,
                seq_len,
                latency,
                roofline_latency,
                gpu_latency,
            )
        )

    # Save results
    df_out = pd.DataFrame(
        latency_list,
        columns=[
            "batch_size",
            "num_heads_q",
            "num_heads_kv",
            "head_dim",
            "seq_len",
            "Ours",
            "Roofline",
            "GPU",
        ],
    )
    df_out.to_csv(file_name, index=False)


def test_and_save_latency_decode(
    test_params: list,
    file_name: str,
    precision: str,
    pcb: Any,
    args: argparse.Namespace,
):
    """
    Tests latency for FlashAttention Decoding phase and saves results to CSV.

    Args:
        test_params: List of tuples (batch_size, num_heads_q, num_heads_kv, head_dim, seq_len_q, seq_len_kv)
        file_name: Output CSV file path
        precision: Data precision
        pcb: Hardware configuration
        args: Command line arguments
    """
    df = None
    if "all" not in args.update:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)

    qkv_data_type, output_data_type, intermediate_data_type = get_data_types(precision)

    latency_list = []
    for idx, (
        batch_size,
        num_heads_q,
        num_heads_kv,
        head_dim,
        seq_len_q,
        seq_len_kv,
    ) in enumerate(test_params):
        print(
            f"problem: batch_size {batch_size} heads_q {num_heads_q} heads_kv {num_heads_kv} head_dim {head_dim} seq_len_q {seq_len_q} seq_len_kv {seq_len_kv}"
        )

        max_seqlen_kv = 4096

        # Initialize Decode Model
        model = FlashAttentionDecode(
            qkv_data_type=qkv_data_type,
            output_data_type=output_data_type,
            intermediate_data_type=intermediate_data_type,
            pcb_module=pcb,
            is_causal=False,
            num_splits=0,
        )

        # Configure model (Decoding usually has Q len=1 or small, KV len grows)
        _ = model(
            Tensor((batch_size, num_heads_q, seq_len_q, head_dim), qkv_data_type),
            Tensor((batch_size, num_heads_kv, 1, head_dim), qkv_data_type),
            Tensor((batch_size, num_heads_kv, 1, head_dim), qkv_data_type),
            cache_seqlens=seq_len_kv - 1,
            max_seqlen_kv=max_seqlen_kv,
        )

        # 1. Analytical Model
        if "ours" in args.update or "all" in args.update or df is None:
            latency = 1000 * model.compile_and_simulate(compile_mode="exhaustive")
        else:
            latency = float(df["Ours"].iloc[idx])

        # 2. Roofline Model
        if "roofline" in args.update or "all" in args.update or df is None:
            roofline_latency = 1000 * model.roofline_model()
        else:
            roofline_latency = float(df["Roofline"].iloc[idx])

        # 3. GPU Measurement
        if "gpu" in args.update or "all" in args.update or df is None:
            try:
                gpu_latency = 1000 * model.run_on_gpu()
            except Exception as e:
                print(f"Warning: GPU profiling failed: {e}")
                gpu_latency = -1
        else:
            gpu_latency = float(df["GPU"].iloc[idx])

        print(
            f"Ours: {latency:.3f} ms, Roofline: {roofline_latency:.3f} ms, GPU: {gpu_latency:.3f} ms"
        )
        latency_list.append(
            (
                batch_size,
                num_heads_q,
                num_heads_kv,
                head_dim,
                seq_len_q,
                seq_len_kv,
                latency,
                roofline_latency,
                gpu_latency,
            )
        )

    # Save results
    df_out = pd.DataFrame(
        latency_list,
        columns=[
            "batch_size",
            "num_heads_q",
            "num_heads_kv",
            "head_dim",
            "seq_len_q",
            "seq_len_kv",
            "Ours",
            "Roofline",
            "GPU",
        ],
    )
    df_out.to_csv(file_name, index=False)


def test_model(model_name: str, args: argparse.Namespace):
    model_shapes = test_model_dict[model_name]

    head_dim = model_shapes["head_dim"]
    num_heads_q = model_shapes["num_attention_heads"]
    num_heads_kv = model_shapes["num_key_value_heads"]

    os.makedirs(
        f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}",
        exist_ok=True,
    )
    file_path = f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/attention_TP.csv"

    if args.mode == "prefill":
        M = [1, 2, 4, 8]
        seq_len_list = [512, 3072]

        test_params = []
        sl_begin = seq_len_list[0]
        sl_end = seq_len_list[-1]
        for batch in M:
            for sl in range(sl_begin, sl_end + 1, 128):
                test_params.append((batch, num_heads_q, num_heads_kv, head_dim, sl))

        test_and_save_latency_prefill(test_params, file_path, args.precision, pcb, args)

    elif args.mode == "decode":
        M = [1, 2, 4]
        seq_len_q = [32, 64]
        seq_len_kv_total = [512, 3072]

        test_params = []
        slk_begin = seq_len_kv_total[0]
        slk_end = seq_len_kv_total[-1]
        for batch in M:
            for slq in seq_len_q:
                if slq * batch > 128 or slq * batch < 32:
                    continue
                for slk in range(slk_begin, slk_end + 1, 128):
                    test_params.append(
                        (batch, num_heads_q, num_heads_kv, head_dim, slq, slk)
                    )

        test_and_save_latency_decode(test_params, file_path, args.precision, pcb, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["Orin", "Thor"], default="Orin")
    parser.add_argument(
        "--mode", type=str, choices=["prefill", "decode"], default="prefill"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "InternVision",
            "Qwen3_0_6B",
            "Qwen3_1_7B",
            "Qwen3_4B",
            "Qwen3_8B",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8", "int4", "all"],
        default="all",
    )
    parser.add_argument(
        "--update",
        nargs="+",
        choices=["ours", "roofline", "gpu", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    pcb = device_dict[args.device]

    if args.model != "all":
        model_list = [args.model]
    else:
        model_list = [
            "InternVision",
            "Qwen3_0_6B",
            "Qwen3_1_7B",
            "Qwen3_4B",
            "Qwen3_8B",
        ]

    if args.precision != "all":
        precision_list = [args.precision]
    else:
        # Defaulting to fp16 if specific precision not selected
        precision_list = ["fp16"]

    for model in model_list:
        args.model = model
        for precision in precision_list:
            args.precision = precision
            test_model(model, args)
