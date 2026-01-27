import json
import os

def load_json_data(file_name="UCIE"):
    file_name = f"configs/{file_name}.json"
    
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

def get_min_ucie_rate(data_size_bytes, expected_time_s, spec_db):
    if expected_time_s <= 0:
        return {"error": "Expected time must be greater than 0"}
    
    expected_time_s -= spec_db["latency"]
    required_bw_bps = data_size_bytes / expected_time_s / spec_db["bandwidth_efficiency"]
    result = {}

    for pkg_key, pkg_info in spec_db["packages"].items():
        lane_count = pkg_info["lane_count"]
        found_rate = None
        for config in pkg_info["configurations"]:
            rate = config["rate_gt"]
            capacity_bps = (rate * 1e9 * lane_count * spec_db["max_modules"]) / 8
            if capacity_bps >= required_bw_bps:
                found_rate = rate
                break
        result[pkg_key] = found_rate
    return result

def calculate_ucie_phy_area(package_type, data_rate_gt, spec_db):
    pkg_key = package_type.lower()
    if pkg_key not in spec_db["packages"]:
        raise ValueError(f"Unknown package type: {package_type}")
    
    pkg_info = spec_db["packages"][pkg_key]
    width_um = pkg_info["phy_width_um"]
    depth_um = None
    
    for config in pkg_info["configurations"]:
        if config["rate_gt"] == data_rate_gt:
            depth_um = config["phy_depth_um"]
            break
            
    if depth_um is None:
        raise ValueError(f"Data rate {data_rate_gt} GT/s not found in JSON for {package_type}")
        
    area_mm2 = (width_um * depth_um) / 1_000_000.0
    return area_mm2 * spec_db["max_modules"]

def calculate_ucie_average_power(package_type, data_rate_gt, voltage, data_size_bytes, expected_time_s, spec_db, multiplier=2):
    pkg_key = package_type.lower()
    if pkg_key not in spec_db["packages"]:
        raise ValueError(f"Package type '{package_type}' not found.")
    
    efficiency_dict = None
    for config in spec_db["packages"][pkg_key]["configurations"]:
        if config["rate_gt"] == data_rate_gt:
            efficiency_dict = config["efficiency_pj_per_bit"]
            break
            
    if efficiency_dict is None:
        raise ValueError(f"Data rate {data_rate_gt} GT/s not found for {package_type}.")
    if voltage not in efficiency_dict:
        raise ValueError(f"Voltage '{voltage}' not defined for this configuration.")
        
    pj_per_bit = efficiency_dict[voltage]
    total_bits = data_size_bytes * 8 / spec_db["bandwidth_efficiency"]
    total_energy_joules = total_bits * (pj_per_bit * 1e-12)
    average_power_watts = total_energy_joules / expected_time_s
    return average_power_watts * multiplier

if __name__ == "__main__":
    ucie_spec = load_json_data()

    data = 200 * 1024**3
    time = 1

    min_rates = get_min_ucie_rate(data, time, ucie_spec)

    print(f"--- Transfer: {data/1024**3} GB / {time} s ---")
    for pkg, rate in min_rates.items():
        if rate:
            area = calculate_ucie_phy_area(pkg, rate, ucie_spec)
            print(f"Package: {pkg:8} | Min rate {rate:2} GT/s | Area {area:.4f} mm²")
        else:
            print(f"Package: {pkg:8} | Requirement cannot be satisfied")

    avg_p = calculate_ucie_average_power(
        package_type="advanced", 
        data_rate_gt=16, 
        voltage="0.5V", 
        data_size_bytes=data, 
        expected_time_s=time,
        spec_db=ucie_spec
    )

    print(f"Average power of advanced package at 0.5V: {avg_p:.4f} W")

    avg_p_high = calculate_ucie_average_power("advanced", 16, "0.7V", data, time, ucie_spec)
    print(f"Average power of advanced package at 0.7V: {avg_p_high:.4f} W")