import json
import os
import argparse

def load_power_data(hardware_name):
    file_name = f"configs/{hardware_name}_power_params.json"
    
    if not os.path.exists(file_name):
        print(f"Error: {file_name} not found")
        return None
        
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
            print(f"Read file success: {file_name}")
            return data
    except json.JSONDecodeError:
        print(f"Error: {file_name} format error")
        return None

def calculate_matmul_power(hw_name, precision, mem_access_bytes, input_bytes, total_fma, runtime):
    """
    Mem Power = mem_intercept + k_mem * mem_access_size
    SoC Power = soc_intercept + k_soc_fma * total_fma + k_soc_input * input_size
    """
    mem_access_bytes /= runtime
    input_bytes /= runtime
    total_fma /= runtime

    power_data = load_power_data(hw_name)
    if not power_data: return None
    
    op_data = power_data['matmul']
    
    # 1. Memory Power Calculation
    k_mem = op_data['k_mem']
    mem_intercept = op_data['mem_intercept']
    mem_power = mem_intercept + (k_mem * mem_access_bytes)
    
    # 2. SoC Power Calculation
    if precision not in op_data['k_soc_fma'] or precision not in op_data['k_soc_input']:
        print(f"Error: Precision '{precision}' not supported in {hw_name}'s power config.")
        return None

    soc_intercept = op_data['soc_intercept']
    k_soc_fma = op_data['k_soc_fma'][precision]
    k_soc_input = op_data['k_soc_input'][precision]
    
    soc_power = soc_intercept + (k_soc_fma * total_fma) + (k_soc_input * input_bytes)
    
    # 3. Summarize
    total_power = mem_power + soc_power
    
    return {
        "operator": "Matmul",
        "config": {
            "hardware": hw_name,
            "precision": precision,
            "mem_access_gb": round(mem_access_bytes / 1e9, 4),
            "input_gb": round(input_bytes / 1e9, 4),
            "total_fma_trillions": round(total_fma / 1e12, 4)
        },
        "power_breakdown_watts": {
            "total": round(total_power, 2),
            "memory_power": round(mem_power, 2),
            "soc_power": round(soc_power, 2)
        }
    }

def calculate_layernorm_power(hw_name, mode, mem_access_bytes, runtime):
    """
    Mem Power = mem_intercept + k_mem * mem_access_size
    SoC Power = fma_intercept (chose by prefill/decode)
    """
    mem_access_bytes /= runtime

    power_data = load_power_data(hw_name)
    if not power_data: return None
    
    op_data = power_data['layernorm']
    
    # 1. Memory Power Calculation
    k_mem = op_data['k_mem']
    mem_intercept = op_data['mem_intercept']
    
    mem_power = mem_intercept + (k_mem * mem_access_bytes)
    
    # 2. SoC Power Calculation
    if mode not in op_data['fma_intercept']:
        print(f"Error: mode '{mode}' not supported (use 'prefill' or 'decode').")
        return None
        
    soc_power = op_data['fma_intercept'][mode]
    
    # 3. Summarize
    total_power = mem_power + soc_power
    
    return {
        "operator": "LayerNorm",
        "config": {
            "hardware": hw_name,
            "mode": mode,
            "mem_access_gb": round(mem_access_bytes / 1e9, 4)
        },
        "power_breakdown_watts": {
            "total": round(total_power, 2),
            "memory_power": round(mem_power, 2),
            "soc_power": round(soc_power, 2)
        }
    }

def print_report(res):
    if not res: return
    print(f"\n--- Power Simulation: {res['operator']} on {res['config']['hardware']} ---")
    
    conf_str = ", ".join([f"{k}={v}" for k, v in res['config'].items() if k != 'hardware'])
    print(f"Inputs          : [{conf_str}]")
    
    print(f"Total Power     : {res['power_breakdown_watts']['total']} W")
    print(f" > Memory Power : {res['power_breakdown_watts']['memory_power']} W")
    print(f" > SoC Power    : {res['power_breakdown_watts']['soc_power']} W")
    
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hw", type=str, choices=["Orin", "Thor"], help="Hardware name")
    args = parser.parse_args()

    if args.hw == "Thor":
        res_matmul_fp8 = calculate_matmul_power(
            args.hw, 
            precision="fp8", 
            mem_access_bytes=(1024+2560)*9728+1024*2560, 
            input_bytes=(1024+2560)*9728, 
            total_fma=1024*2560*9728,
            runtime=0.272e-3
        )
        print_report(res_matmul_fp8)
        
        res_ln_prefill = calculate_layernorm_power(
            args.hw,
            mode="prefill",
            mem_access_bytes=1024*4096,
            runtime=0.088e-3
        )
        print_report(res_ln_prefill)

        res_ln_decode = calculate_layernorm_power(
            args.hw,
            mode="decode",
            mem_access_bytes=64*4096,
            runtime=0.0055e-3
        )
        print_report(res_ln_decode)

    elif args.hw == "Orin":
        res_matmul_fp8 = calculate_matmul_power(
            args.hw, 
            precision="int8", 
            mem_access_bytes=(1024+2560)*9728+1024*2560, 
            input_bytes=(1024+2560)*9728, 
            total_fma=1024*2560*9728,
            runtime=0.68e-3
        )
        print_report(res_matmul_fp8)
    else:
        print("Hardware config not ready.")