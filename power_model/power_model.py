import argparse
import json
import os

from hardware_model.device import device_dict
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
from test.matmul.utils import get_output_dtype

_POWER_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_power_data(hardware_name):
    file_name = os.path.join(
        _POWER_MODEL_DIR, "configs", f"{hardware_name}_power_params.json"
    )

    if not os.path.exists(file_name):
        print(f"Error: {file_name} not found")
        return None

    try:
        with open(file_name, "r") as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError:
        print(f"Error: {file_name} format error")
        return None


def _sm_scale(hw_name, pcb):
    """Return soc_intercept scaling factor based on actual vs. reference SM count."""
    if pcb is None:
        return 1.0
    ref_sm = device_dict[hw_name].compute_module.core_count
    return pcb.compute_module.core_count / ref_sm


def calculate_matmul_power(
    hw_name, precision, mem_access_bytes, total_fma, runtime, pcb=None
):
    mem_access_bytes /= runtime
    total_fma /= runtime

    power_data = load_power_data(hw_name)
    if not power_data:
        return None

    k_mem = power_data["k_mem"]
    mem_intercept = power_data["intercept"]["mem"]
    mem_power = mem_intercept + (k_mem * mem_access_bytes)

    soc_intercept_map = power_data["intercept"]["soc"]["matmul"]
    k_soc_fma_map = power_data["k_soc"]["fma"]["matmul"]
    dram_map = power_data["k_soc"]["dram_access"]["matmul"]

    if precision not in soc_intercept_map or precision not in k_soc_fma_map:
        print(
            f"Error: Precision '{precision}' not supported in {hw_name}'s power config."
        )
        return None

    soc_intercept = soc_intercept_map[precision] * _sm_scale(hw_name, pcb)
    k_soc_fma = k_soc_fma_map[precision]
    k_soc_dram = dram_map.get(precision, 0)

    soc_power = (
        soc_intercept + (k_soc_fma * total_fma) + (k_soc_dram * mem_access_bytes)
    )
    total_power = mem_power + soc_power

    return {
        "operator": "Matmul",
        "config": {
            "hardware": hw_name,
            "precision": precision,
            "mem_access_gb": round(mem_access_bytes / 1e9, 4),
            "total_fma_trillions": round(total_fma / 1e12, 4),
        },
        "power_breakdown_watts": {
            "total": round(total_power, 2),
            "memory_power": round(mem_power, 2),
            "soc_power": round(soc_power, 2),
        },
    }


def calculate_layernorm_power(
    hw_name, mode, mem_access_bytes, runtime, precision="fp16", pcb=None
):
    mem_access_bytes /= runtime

    power_data = load_power_data(hw_name)
    if not power_data:
        return None

    k_mem = power_data["k_mem"]
    mem_intercept = power_data["intercept"]["mem"]
    mem_power = mem_intercept + (k_mem * mem_access_bytes)

    soc_intercept_map = power_data["intercept"]["soc"]["layernorm"]
    dram_map = power_data["k_soc"]["dram_access"]["layernorm"]

    if precision not in soc_intercept_map:
        print(
            f"Error: precision '{precision}' not supported for layernorm in {hw_name}."
        )
        return None

    soc_intercept = soc_intercept_map[precision] * _sm_scale(hw_name, pcb)
    k_soc_dram = dram_map.get(precision, 0)
    soc_power = soc_intercept + (k_soc_dram * mem_access_bytes)
    total_power = mem_power + soc_power

    return {
        "operator": "LayerNorm",
        "config": {
            "hardware": hw_name,
            "mode": mode,
            "precision": precision,
            "mem_access_gb": round(mem_access_bytes / 1e9, 4),
        },
        "power_breakdown_watts": {
            "total": round(total_power, 2),
            "memory_power": round(mem_power, 2),
            "soc_power": round(soc_power, 2),
        },
    }


def calculate_flashattn_power(
    hw_name, precision, mem_access_bytes, fma_count, runtime, pcb=None
):
    mem_access_bytes /= runtime
    fma_count /= runtime

    power_data = load_power_data(hw_name)
    if not power_data:
        return None

    k_mem = power_data["k_mem"]
    mem_intercept = power_data["intercept"]["mem"]
    mem_power = mem_intercept + k_mem * mem_access_bytes

    soc_intercept_map = power_data["intercept"]["soc"]["flashattn"]
    k_soc_fma_map = power_data["k_soc"]["fma"]["flashattn"]
    dram_map = power_data["k_soc"]["dram_access"]["flashattn"]

    if precision not in soc_intercept_map or precision not in k_soc_fma_map:
        print(
            f"Error: Precision '{precision}' not supported for flashattn in {hw_name}'s power config."
        )
        return None

    soc_intercept = soc_intercept_map[precision] * _sm_scale(hw_name, pcb)
    k_soc_fma = k_soc_fma_map[precision]
    k_soc_dram = dram_map.get(precision, 0)

    soc_power = soc_intercept + k_soc_fma * fma_count + k_soc_dram * mem_access_bytes
    total_power = mem_power + soc_power

    return {
        "operator": "FlashAttn",
        "config": {
            "hardware": hw_name,
            "precision": precision,
            "mem_access_gb": round(mem_access_bytes / 1e9, 4),
            "total_fma_trillions": round(fma_count / 1e12, 4),
        },
        "power_breakdown_watts": {
            "total": round(total_power, 2),
            "memory_power": round(mem_power, 2),
            "soc_power": round(soc_power, 2),
        },
    }


def print_report(res):
    if not res:
        return
    print(
        f"\n--- Power Simulation: {res['operator']} on {res['config']['hardware']} ---"
    )

    conf_str = ", ".join(
        [f"{k}={v}" for k, v in res["config"].items() if k != "hardware"]
    )
    print(f"Inputs          : [{conf_str}]")

    print(f"Total Power     : {res['power_breakdown_watts']['total']} W")
    print(f" > Memory Power : {res['power_breakdown_watts']['memory_power']} W")
    print(f" > SoC Power    : {res['power_breakdown_watts']['soc_power']} W")

    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device", type=str, choices=["Orin", "Thor"], help="Hardware name"
    )
    args = parser.parse_args()

    if args.device == "Thor":
        # Test matmul
        M = 1024
        N = 2560
        K = 9728
        pcb = device_dict[args.device]
        activation_dtype = data_type_dict["fp8"]
        weight_dtype = data_type_dict["fp8"]
        intermediate_dtype = data_type_dict["fp32"]
        output_dtype = get_output_dtype(activation_dtype, "down_proj", True)
        model = Matmul(
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            intermediate_dtype=intermediate_dtype,
            output_dtype=output_dtype,
            device=args.device,
        )
        _ = model(
            Tensor([M, K], dtype=activation_dtype),
            Tensor([K, N], dtype=weight_dtype),
        )
        latency = (
            model.compile_and_simulate(pcb) + pcb.compute_module.launch_latency.matmul
        )
        res_matmul_fp8 = calculate_matmul_power(
            args.device,
            precision="fp8",
            mem_access_bytes=model.mem_access_size,
            total_fma=model.fma_count,
            runtime=latency,
        )
        print_report(res_matmul_fp8)
