import json
import os


def load_json_data(file_name):
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


def get_nearest_ucie_configurations(expected_bw_gbps, spec_db):
    if expected_bw_gbps <= 0:
        return {"error": "Expected bw must be greater than 0"}

    result = {}

    for pkg_key, pkg_info in spec_db["packages"].items():
        nearest = []
        nearest_bandwidth_gap = float("inf")
        for module_count in spec_db["available_module_count"]:
            for lane_count in pkg_info["available_lane_count"]:
                for config in pkg_info["configurations"]:
                    rate_gt = config["rate_gt"]
                    capacity_gbps = rate_gt * lane_count * module_count
                    bandwidth_gap = abs(capacity_gbps - expected_bw_gbps)
                    if bandwidth_gap < nearest_bandwidth_gap:
                        nearest = [
                            {
                                "module_count": module_count,
                                "lane_count": lane_count,
                                "rate_gt": rate_gt,
                            }
                        ]
                        nearest_bandwidth_gap = bandwidth_gap
                    elif bandwidth_gap == nearest_bandwidth_gap:
                        nearest.append(
                            {
                                "module_count": module_count,
                                "lane_count": lane_count,
                                "rate_gt": rate_gt,
                            }
                        )
        result[pkg_key] = sorted(
            nearest, key=lambda x: (x["module_count"], x["lane_count"], x["rate_gt"])
        )
    return result


def calculate_ucie_phy_area(package_type, module_count, lane_count, rate_gt, spec_db):
    pkg_key = package_type.lower()

    pkg_info = spec_db["packages"][pkg_key]
    width_um = pkg_info["phy_width_um"]

    for config in pkg_info["configurations"]:
        if config["rate_gt"] == rate_gt:
            depth_um = config["phy_depth_um"]
            if lane_count == 32:
                depth_um *= pkg_info["x32_phy_depth_discount"]
            break

    area_mm2 = (width_um * depth_um) / 1_000_000.0
    return area_mm2 * module_count


def calculate_ucie_dynamic_energy(
    package_type,
    rate,
    voltage,
    data_size_bytes,
    spec_db,
):
    pkg_key = package_type.lower()
    if pkg_key not in spec_db["packages"]:
        raise ValueError(f"Package type '{package_type}' not found.")

    efficiency_dict = None
    for config in spec_db["packages"][pkg_key]["configurations"]:
        if config["rate_gt"] == rate:
            efficiency_dict = config["efficiency_pj_per_bit"]
            break

    if efficiency_dict is None:
        raise ValueError(f"Data rate {rate} GT/s not found for {package_type}.")
    if voltage not in efficiency_dict:
        raise ValueError(f"Voltage '{voltage}' not defined for this configuration.")

    pj_per_bit = efficiency_dict[voltage]
    total_bits = data_size_bytes * 8
    total_energy_mj = total_bits * (pj_per_bit * 1e-9)
    return total_energy_mj


def get_nearest_pcie_lane(expected_bw_gbps, spec_db):
    if expected_bw_gbps <= 0:
        return {"error": "Expected bw must be greater than 0"}

    rate_gt = spec_db["rate_gt"]

    nearest = None
    nearest_bandwidth_gap = float("inf")
    for lane_count in spec_db["available_lane_count"]:
        capacity_gbps = rate_gt * lane_count
        bandwidth_gap = abs(capacity_gbps - expected_bw_gbps)
        if bandwidth_gap < nearest_bandwidth_gap:
            nearest = lane_count
            nearest_bandwidth_gap = bandwidth_gap
    return nearest


def calculate_pcie_dynamic_energy(data_size_bytes, spec_db):
    pj_per_bit = spec_db["efficiency_pj_per_bit"]
    total_bits = data_size_bytes * 8
    total_energy_mj = total_bits * (pj_per_bit * 1e-9)
    return total_energy_mj


if __name__ == "__main__":
    ucie_spec = load_json_data("configs/UCIE.json")
    if ucie_spec is None:
        raise SystemExit(1)

    data_size_gigabytes = 32
    time_s = 1.0
    expected_bw_gbps = data_size_gigabytes * 8 / time_s

    ucie_nearest = get_nearest_ucie_configurations(expected_bw_gbps, ucie_spec)

    print(
        f"--- Transfer: {data_size_gigabytes} GB / {time_s} s (~{expected_bw_gbps:.1f} Gbps) ---"
    )
    for pkg, candidates in ucie_nearest.items():
        for candidate in candidates:
            module_count, lane_count, rate_gt = (
                candidate["module_count"],
                candidate["lane_count"],
                candidate["rate_gt"],
            )
            capacity_gbps = module_count * lane_count * rate_gt
            area = calculate_ucie_phy_area(
                pkg, module_count, lane_count, rate_gt, ucie_spec
            )
            energy_0_7v_mj = calculate_ucie_dynamic_energy(
                pkg, rate_gt, "0.7V", data_size_gigabytes * 1024**3, ucie_spec
            )
            energy_0_5v_mj = calculate_ucie_dynamic_energy(
                pkg, rate_gt, "0.5V", data_size_gigabytes * 1024**3, ucie_spec
            )
            power_0_7v = energy_0_7v_mj * 1e-3 / time_s
            power_0_5v = energy_0_5v_mj * 1e-3 / time_s
            print(
                f"{pkg:8} | {module_count} module(s) × {lane_count} lane(s) x {rate_gt} GT/s = {capacity_gbps:.1f} Gbps | Area {area:.4f} mm² | Power 0.7V: {power_0_7v:.4f} W, 0.5V: {power_0_5v:.4f} W"
            )

    print()

    pcie_spec = load_json_data("configs/PCIE.json")
    if pcie_spec is None:
        raise SystemExit(1)
    pcie_lanes = get_nearest_pcie_lane(expected_bw_gbps, pcie_spec)
    print(
        f"Nearest PCIe lane count: {pcie_lanes}, bandwidth {pcie_lanes * pcie_spec['rate_gt']:.1f} Gbps"
    )
    energy_pcie_mj = calculate_pcie_dynamic_energy(
        data_size_gigabytes * 1024**3, pcie_spec
    )
    avg_p_pcie = energy_pcie_mj * 1000 / time_s
    avg_p_pcie += (
        pcie_spec["switch_power_k"] * pcie_lanes + pcie_spec["switch_power_intercept"]
    )
    print(f"Average power of PCIe: {avg_p_pcie:.4f} W")
