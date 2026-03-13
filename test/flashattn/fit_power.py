import argparse
import json
import os
from math import inf

import numpy as np

from hardware_model.device import device_dict
from software_model.flashattn import FlashAttn
from software_model.flashattn_combine import FlashAttentionCombine
from software_model.utils import Tensor, data_type_dict
from test.flashattn.utils import get_output_dtype
from test.utils import fit_single_rail, plot_fitting_results, print_rail_results

file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE_TEMPLATE = f"{file_dir}/temp/power_features_cache"

intercept_dict = {"Orin": {"soc": 25, "mem": 0.5}, "Thor": {"soc": 25, "mem": 6.7}}


def load_or_generate_data(args):
    cache_path = f"{CACHE_FILE_TEMPLATE}_{args.precision}.{args.device}.npz"
    X, y_soc, y_mem = None, None, None

    if os.path.exists(cache_path):
        print(f"Found cache file: {cache_path}")
        try:
            data = np.load(cache_path)
            if "y_soc" in data and "y_mem" in data:
                X = data["X"]
                y_soc = data["y_soc"]
                y_mem = data["y_mem"]
                print(f"Loaded {len(X)} records from cache.")
                return X, y_soc, y_mem
        except Exception as e:
            print(f"Error loading cache: {e}. Will re-calculate.")

    print("Generating features from simulation...")
    pcb = device_dict[args.device]
    existing_data = []

    json_path = f"{file_dir}/temp/power_log.{args.device}.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            content = f.read().strip()
            if content:
                existing_data = json.loads(content)
    else:
        raise FileNotFoundError(f"{json_path} not found")

    X_features = []
    y_soc_list = []
    y_mem_list = []

    for record in existing_data:
        (
            seq_len_q,
            seq_len_kv,
            num_heads_q,
            num_heads_kv,
            head_dim,
            is_causal,
            precision,
            output_dtype_name,
        ) = (
            record["seq_len_q"],
            record["seq_len_kv"],
            record["num_heads_q"],
            record["num_heads_kv"],
            record["head_dim"],
            record["is_causal"],
            record["precision"],
            record["output_dtype"],
        )
        is_prefill = seq_len_q == seq_len_kv
        if precision != args.precision:
            continue
        assert precision == output_dtype_name
        if precision == "fp16":
            qkv_dtype = data_type_dict["fp16"]
            intermediate_dtype = data_type_dict["fp32"]
            assert args.device in ("Orin", "Thor"), (
                "fp16 precision is for Orin and Thor"
            )
        elif precision == "fp8":
            qkv_dtype = data_type_dict["fp8"]
            intermediate_dtype = data_type_dict["fp32"]
            assert args.device == "Thor", "fp8 precision is for Thor only"
        else:
            raise ValueError("Unsupported precision")
        output_dtype = get_output_dtype(data_type_dict[precision], True)

        best_latency = inf
        best_model = None
        best_model1 = None
        num_splits_list = [1] if is_prefill else [1, 2, 4]
        for num_splits in num_splits_list:
            model = FlashAttn(
                qkv_dtype,
                intermediate_dtype,
                output_dtype,
                is_prefill,
                is_causal,
                num_splits,
                args.device,
            )
            _ = model(
                Tensor([seq_len_q, num_heads_q, head_dim], dtype=qkv_dtype),
                Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
                Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
            )
            launch_latency = (
                pcb.compute_module.launch_latency.flashattn_prefill
                if is_prefill
                else pcb.compute_module.launch_latency.flashattn_decode
            )
            latency_this = 1000 * (model.compile_and_simulate(pcb) + launch_latency)
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
            best_latency = min(best_latency, latency_this)
            best_model = model
            if num_splits > 1:
                best_model1 = model1

        runtime_s = best_latency / 1000.0

        features = [
            best_model.fma_count / runtime_s,  # 0: FMA
            best_model.head_dim
            * (
                best_model.num_heads_q * best_model.seq_len_q  # Q
                + best_model.num_heads_kv * best_model.seq_len_kv * 2  # K & V
            )
            * best_model.qkv_dtype.word_size
            / runtime_s,  # 1: Input
            best_model.head_dim
            * best_model.num_heads_q
            * best_model.seq_len_q
            * best_model.output_dtype.word_size
            / runtime_s,  # 2. Output
            best_model.mem_access_size / runtime_s,  # 3: DRAM
        ]

        X_features.append(features)
        y_soc_list.append(record["power_GPU"])
        y_mem_list.append(record["power_MEM"])

        print(
            f"seq_len_q: {seq_len_q}, seq_len_kv: {seq_len_kv}, num_heads_q: {num_heads_q}, num_heads_kv: {num_heads_kv}, head_dim: {head_dim} | Latency={best_latency:.2f}ms | SOC={record['power_GPU']}W, MEM={record['power_MEM']}W"
        )
        print(f"  Features_raw: FMA={model.fma_count}, DRAM={model.mem_access_size}")

    if len(X_features) > 0:
        X = np.array(X_features)
        y_soc = np.array(y_soc_list)
        y_mem = np.array(y_mem_list)

        print(f"Saving new cache to: {cache_path}")
        np.savez(cache_path, X=X, y_soc=y_soc, y_mem=y_mem)
    else:
        print("No valid data generated.")
        return None, None, None

    return X, y_soc, y_mem


def fit_and_analyze_rails(X_raw, y_soc, y_mem, args):
    full_feature_names = [
        "FMA",
        "INPUT Size",
        "OUTPUT Size",
        "DRAM Access Byte",
    ]
    # full_feature_names = ["FMA", "DRAM Access Byte"]

    feat_map = {name: i for i, name in enumerate(full_feature_names)}

    # Custom features here
    # ==============================================================================
    soc_features_to_use = ["FMA", "DRAM Access Byte"]
    # Not using OUTPUT Size, since it colinear with FMA!

    mem_features_to_use = ["DRAM Access Byte"]
    # ==============================================================================

    print("\n" + "=" * 80)
    print(" DUAL RAIL POWER MODELING (Configurable Feature Subsets) ")
    print("=" * 80)

    res_soc = fit_single_rail(
        X_raw,
        y_soc,
        soc_features_to_use,
        "Rail 1: GPU",
        feat_map,
        full_feature_names,
        enforce_positive=True,
    )
    res_mem = fit_single_rail(
        X_raw,
        y_mem - intercept_dict[args.device]["mem"],
        mem_features_to_use,
        "Rail 2: MEM",
        feat_map,
        full_feature_names,
        enforce_positive=True,
        fit_intercept=False,
    )

    print_rail_results("SoC", res_soc, full_feature_names, soc_features_to_use)

    print_rail_results("Mem", res_mem, full_feature_names, mem_features_to_use)

    plot_fitting_results(
        y_soc,
        res_soc["y_pred"],
        full_feature_names,
        res_soc["coefs"],
        res_soc["intercept"],
        res_soc["r2"],
        res_soc["mape"],
        f"{file_dir}/results_power",
        title_suffix=f"soc_{args.precision}_{args.device}",
    )

    plot_fitting_results(
        y_mem,
        res_mem["y_pred"] + intercept_dict[args.device]["mem"],
        full_feature_names,
        res_mem["coefs"],
        res_mem["intercept"] + intercept_dict[args.device]["mem"],
        res_mem["r2"],
        res_mem["mape"],
        f"{file_dir}/results_power",
        title_suffix=f"mem_{args.precision}_{args.device}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=["Orin", "Thor"])
    parser.add_argument(
        "precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()

    X, y_soc, y_mem = load_or_generate_data(args)

    if X is not None and len(X) > 0:
        fit_and_analyze_rails(X, y_soc, y_mem, args)
    else:
        print("Exiting: No data available.")
