import json
import os
import argparse

def load_area_data(hardware_name):
    file_name = f"configs/{hardware_name}_die_mm.json"
    
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

def calculate_gpu_area(hw, target_sm_count, target_l2_size_mb, target_l1_size_kb=None, target_reg_size_kb=None):
    """    
    Returns:
        dict: include total die area and breakdown
    """
    gpu_data = load_area_data(hw)
    # 1. Prepare Base Data

    # Command Front (fixed)
    area_cmd_front = gpu_data['command_front_area']
    
    # L2 Cache
    l2_info = gpu_data['l2_cache']
    area_l2_ctrl = l2_info['ctrl'] # Controller (fixed)
    base_l2_data_area_total = l2_info['data']['unit_count'] * l2_info['data']['area_per_unit']
    l2_data_area_per_mb = base_l2_data_area_total / l2_info['capacity_mb']

    # SM (single SM area)
    sm_info = gpu_data['sm']
    base_sm_count = sm_info['unit_count']
    
    # L1 and Reg area within SM
    base_l1_kb = sm_info['l1_capacity_kb']
    base_reg_kb = sm_info['reg_capacity_kb']
    area_reg_l1_base = sm_info['area_breakdown']['reg_and_l1']
    area_core_fixed = sm_info['area_breakdown']['core']
    area_others_fixed = sm_info['area_breakdown']['others']
    reg_l1_density = area_reg_l1_base / (base_l1_kb + base_reg_kb)
    t_l1_kb = target_l1_size_kb if target_l1_size_kb is not None else base_l1_kb
    t_reg_kb = target_reg_size_kb if target_reg_size_kb is not None else base_reg_kb

    # GPC/TPC Extra (apportion among SMs)
    gpc_extra = gpu_data['gpc_extra']
    tpc_extra = gpu_data['tpc_extra']
    base_gpc_extra_total = gpc_extra['unit_count'] * gpc_extra['area_per_unit']
    base_tpc_extra_total = tpc_extra['unit_count'] * tpc_extra['area_per_unit']
    extra_per_sm = (base_gpc_extra_total + base_tpc_extra_total) / base_sm_count

    # 2. Target Calculation
    
    # SM
    new_reg_l1_area = (t_l1_kb + t_reg_kb) * reg_l1_density
    new_single_sm_area = new_reg_l1_area + area_core_fixed + area_others_fixed
    new_sm_total_area = target_sm_count * new_single_sm_area
    
    # SM extra
    new_extra_area = target_sm_count * extra_per_sm
    
    # L2
    new_l2_data_area = target_l2_size_mb * l2_data_area_per_mb
    new_l2_total_area = new_l2_data_area + area_l2_ctrl
    
    # 3. summarize
    total_die_area = (
        area_cmd_front + 
        new_l2_total_area + 
        new_sm_total_area + 
        new_extra_area
    )
    
    return {
        "config": {
            "sm_count": target_sm_count,
            "l2_size_mb": target_l2_size_mb,
            "sm_l1_kb": t_l1_kb,
            "sm_reg_kb": t_reg_kb
        },
        "area_breakdown_mm2": {
            "total": round(total_die_area, 2),
            "command_front": round(area_cmd_front, 2),
            "l2_total": round(new_l2_total_area, 2),
            "sm_total": round(new_sm_total_area, 2),
            "extra_gpc_tpc": round(new_extra_area, 2)
        }
    }

def print_report(res):
    if not res: return
    print(f"\n--- Custom GPU Config: {res['config']['sm_count']} SMs, {res['config']['l2_size_mb']}MB L2, {res['config']['sm_l1_kb']}KB L1 ---")
    print(f"Total Die Area  : {res['area_breakdown_mm2']['total']} mm²")
    print(f" > Command Front: {res['area_breakdown_mm2']['command_front']} mm² (Fixed)")
    print(f" > L2 Cache     : {res['area_breakdown_mm2']['l2_total']} mm² (Scaled Data + Fixed Ctrl)")
    print(f" > SM Array     : {res['area_breakdown_mm2']['sm_total']} mm² (Scaled L1 + Reg + Fixed Core/Others)")
    print(f" > GPC/TPC Extra: {res['area_breakdown_mm2']['extra_gpc_tpc']} mm² (Proportional to SM)")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hw", type=str, choices=["Orin", "Thor"], help="Hardware name")
    args = parser.parse_args()
    
    if args.hw == "Thor":
        res_thor = calculate_gpu_area(args.hw, target_sm_count=22, target_l2_size_mb=24)
        print_report(res_thor)
        res_thor_half = calculate_gpu_area(args.hw, target_sm_count=10, target_l2_size_mb=12)
        print_report(res_thor_half)
    elif args.hw == "Orin":
        res_orin = calculate_gpu_area(args.hw, target_sm_count=16, target_l2_size_mb=4.0)
        print_report(res_orin)

        res_orin_half = calculate_gpu_area(args.hw, target_sm_count=8, target_l2_size_mb=2)
        print_report(res_orin_half)
    else:
        assert False