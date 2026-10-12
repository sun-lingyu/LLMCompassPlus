"""First-order Orin/Thor-derived SoC area model (all areas in mm^2).

Run with ``python -m area_model.soc_model --help`` from the repository root.
No performance, power, cost, or DSE assumptions are inferred from this model.
"""

import argparse
import json
import math
from pathlib import Path

from .area_model import calculate_gpu_area, load_area_data

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "Thor_soc.json"
CPU_POLICIES = {"proportional": 0.0, "balanced": 0.5, "fixed": 1.0}


def _number(name, value, *, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    if integer and value != int(value):
        raise ValueError(f"{name} must be an integer")
    return value


def load_soc_data(config_path=None, hw=None):
    """Load an explicit, source-traceable calibration; never depend on cwd."""
    if hw is not None and hw not in {"Thor", "Orin"}:
        raise ValueError("hw must be Thor or Orin")
    path = (
        Path(config_path)
        if config_path is not None
        else CONFIG_PATH.with_name(f"{hw or 'Thor'}_soc.json")
    )
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    hardware = data["reference"].get("hardware", "Thor")
    if hardware not in {"Thor", "Orin"} or (hw is not None and hw != hardware):
        raise ValueError("Hardware selector and SoC calibration disagree")
    data["reference"]["hardware"] = hardware
    return data


def _reconcile_gpu_region(components):
    """Count the complete GPU envelope once, retaining unparameterized support."""
    if "gpu_region" not in components:
        return
    other = components.pop("gpu_region") - components["gpu_compute_complex"]
    if other < 0:
        raise ValueError("GPU outer region is smaller than reference compute complex")
    components["gpu_region_other_fixed"] = other


def _reference_components(data):
    """Use non-overlapping image rectangles for non-GPU components."""
    source = data["measurement"]
    x0, y0, x1, y1 = source["die_bounds_px"]
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid die bounds")
    ref = data["reference"]
    die_area = _number("die_width_mm", ref["die_width_mm"], positive=True) * _number(
        "die_height_mm", ref["die_height_mm"], positive=True
    )
    scale = die_area / ((x1 - x0) * (y1 - y0))
    gpu = calculate_gpu_area(
        ref["hardware"], ref["physical_sm_count"], ref["l2_size_mb"]
    )["area_breakdown_mm2"]["total"]
    result = {"gpu_compute_complex": gpu}
    rectangles = []
    for name, item in source["components"].items():
        area = 0.0
        for box in item["rectangles_px"]:
            left, top, right, bottom = box
            if not (x0 <= left < right <= x1 and y0 <= top < bottom <= y1):
                raise ValueError(f"Invalid rectangle for {name}: {box}")
            for other_name, (a, b, c, d) in rectangles:
                if min(right, c) > max(left, a) and min(bottom, d) > max(top, b):
                    raise ValueError(f"Overlapping rectangles: {name}, {other_name}")
            rectangles.append((name, box))
            area += (right - left) * (bottom - top) * scale
        result[name] = area
    _reconcile_gpu_region(result)
    return die_area, result


def calibrate_overhead(config_path=None, hw=None):
    """Include display/media in the reference before deriving residual overhead.

    Closure is an accounting identity, NOT independent accuracy validation.
    """
    data = load_soc_data(config_path, hw=hw)
    gpu = load_area_data(data["reference"]["hardware"])
    ref = data["reference"]
    if (
        ref["physical_sm_count"] != gpu["sm"]["unit_count"]
        or ref["l2_size_mb"] != gpu["l2_cache"]["capacity_mb"]
        or ref["die_width_mm"] != gpu["width"]
        or ref["die_height_mm"] != gpu["height"]
    ):
        raise ValueError("GPU and SoC reference calibrations disagree")
    die_area, components = _reference_components(data)
    modeled = sum(components.values())
    if not 0 < modeled <= die_area:
        raise ValueError("Reference component sum must be positive and <= die area")
    return {
        "reference_die_mm2": die_area,
        "reference_components_mm2": components,
        "reference_modeled_mm2": modeled,
        "reference_residual_mm2": die_area - modeled,
        "overhead_ratio": (die_area - modeled) / modeled,
    }


def _cpu_configuration(data, active, policy, override, quantum):
    if policy not in CPU_POLICIES:
        raise ValueError(f"cpu_policy must be one of {tuple(CPU_POLICIES)}")
    ref = data["reference"]
    quantum = data["cpu"]["default_core_quantum"] if quantum is None else quantum
    _number("cpu_core_quantum", quantum, positive=True, integer=True)
    beta = CPU_POLICIES[policy]
    requested = ref["cpu_core_count"] * (
        beta + (1 - beta) * active / ref["active_sm_count"]
    )
    if override is None:
        # Positive half-up rounding, not Python round()'s ties-to-even.
        units = requested / quantum
        rounded = math.ceil(units) if policy == "fixed" else math.floor(units + 0.5)
        cores = max(quantum, rounded * quantum)
    else:
        _number("cpu_core_count", override, positive=True, integer=True)
        if override % quantum:
            raise ValueError("cpu_core_count must be a multiple of cpu_core_quantum")
        cores = override
    return cores, quantum, requested


def _validate_link_area(ports, area):
    _number("d2d_port_count", ports, integer=True)
    if ports:
        if area is None:
            raise ValueError("Specify PHY area per D2D port")
        _number("d2d_area_per_port_mm2", area, positive=True)
    elif area is not None:
        raise ValueError("D2D area supplied but d2d_port_count is zero")


def check_memory_phy_shoreline(
    soc_area_mm2, required_length_mm, integration, edge_count=2
):
    """Apply an aggregate square-die edge budget to package integration."""
    _number("soc_area_mm2", soc_area_mm2, positive=True)
    _number("memory_phy_total_length_mm", required_length_mm, positive=True)
    if integration not in {"board", "package"}:
        raise ValueError("integration must be board or package")
    _number("phy_edge_count", edge_count, positive=True, integer=True)
    if edge_count not in (2, 4):
        raise ValueError("phy_edge_count must be 2 (MicroPod) or 4 (monolithic)")
    available = edge_count * math.sqrt(soc_area_mm2)
    fits = required_length_mm <= available or math.isclose(
        required_length_mm, available, rel_tol=1e-12, abs_tol=1e-12
    )
    enforced = integration == "package"
    return {
        "enforced": enforced,
        "edge_count": edge_count,
        "passed": fits if enforced else None,
        "required_length_mm": required_length_mm,
        "available_length_mm": available,
        "margin_mm": available - required_length_mm,
        "minimum_die_area_mm2": (required_length_mm / edge_count) ** 2,
        "reason": (
            f"Memory PHY total length exceeds {edge_count} edges of the equivalent square die"
            if enforced and not fits
            else None
        ),
    }


def _memory_phy_length(data, memory_bus_bits, override):
    if override is not None:
        _number("memory_phy_total_length_mm", override, positive=True)
        return override, "explicit"
    measurement = data["measurement"]
    bounds = measurement["die_bounds_px"]
    boxes = measurement["components"]["memory_phy"]["rectangles_px"]
    axes = data["memory"].get("shoreline_axes", ["y"] * len(boxes))
    if len(axes) != len(boxes) or any(a not in {"x", "y"} for a in axes):
        raise ValueError("Memory shoreline axes must match PHY rectangles")
    width = measurement.get("image_die_width_mm", data["reference"]["die_width_mm"])
    height = measurement.get("image_die_height_mm", data["reference"]["die_height_mm"])
    reference_length = sum(
        (right - left) * width / (bounds[2] - bounds[0])
        if axis == "x"
        else (bottom - top) * height / (bounds[3] - bounds[1])
        for (left, top, right, bottom), axis in zip(boxes, axes)
    )
    return (
        reference_length * memory_bus_bits / data["reference"]["memory_bus_bits"],
        "reference_phy_strips_scaled_by_bus_width",
    )


def _horizontal_phy_length(data, name):
    """Bottom-edge PHY rectangles in the Thor/Orin calibrations, in mm."""
    measurement = data["measurement"]
    bounds = measurement["die_bounds_px"]
    width = measurement.get("image_die_width_mm", data["reference"]["die_width_mm"])
    return (
        sum(box[2] - box[0] for box in measurement["components"][name]["rectangles_px"])
        * width
        / (bounds[2] - bounds[0])
    )


def _validate_extra_phy_inputs(integration, ports, length, extra):
    _number("extra_hardware_area_mm2", extra)
    if length is not None:
        _number("d2d_phy_length_per_port_mm", length, positive=True)
        if not ports:
            raise ValueError("D2D length supplied without physical ports")
    if integration == "package" and ports and length is None:
        raise ValueError("Package integration requires D2D PHY length per port")


def _apply_optional_components(components, data, include_display, include_dla):
    defaults = data.get("defaults", {})
    retained_dla = (
        defaults.get("include_dla", False) if include_dla is None else include_dla
    )
    if not isinstance(retained_dla, bool):
        raise ValueError("include_dla must be a boolean or None")
    if retained_dla and "dla" not in components:
        raise ValueError("This calibration has no separately identified DLA")
    if "dla" in components and not retained_dla:
        components["dla"] = 0.0
    if not include_display:
        if not defaults.get("display_exclusion_supported", True):
            raise ValueError(
                "Display control is not separately identified for this hardware"
            )
        components["display_subsystem"] = 0.0
        components["display_phy"] = 0.0
    return retained_dla


def _apply_vision_accelerator(components, data, include_vision_accelerator):
    retained_vision = (
        data.get("defaults", {}).get("include_vision_accelerator", False)
        if include_vision_accelerator is None
        else include_vision_accelerator
    )
    if not isinstance(retained_vision, bool):
        raise ValueError("include_vision_accelerator must be a boolean or None")
    vision_module = data["vision_accelerator_component"]
    if vision_module not in components:
        raise ValueError("Vision accelerator component is missing from calibration")
    if not retained_vision:
        components[vision_module] = 0.0
    return retained_vision, vision_module


def calculate_soc_area(
    target_sm_count,
    target_l2_size_mb,
    *,
    hw=None,
    active_sm_count=None,
    cpu_policy="balanced",
    cpu_core_count=None,
    cpu_core_quantum=None,
    memory_bus_bits=128,
    integration="board",
    phy_edge_count=2,
    memory_phy_total_length_mm=None,
    include_display=True,
    include_dla=None,
    include_vision_accelerator=None,
    d2d_port_count=0,
    d2d_area_per_port_mm2=None,
    d2d_phy_length_per_port_mm=None,
    extra_hardware_area_mm2=0.0,
    pcie_x8_count=1,
    memory_controller_area_mm2=None,
    overhead_ratio=None,
    target_l1_size_kb=None,
    target_reg_size_kb=None,
    config_path=None,
):
    """Estimate one physical die, including inactive but manufactured resources.

    target_sm_count is PHYSICAL SM count. CPU policy uses active SMs / reference active SMs;
    active defaults to physical for custom designs. CPU quantum is a design
    assumption, not a claim about physical cluster topology. CPU caches scale with
    the rounded core count; L3 is not independently rounded. Scaled policies use
    nearest-quantum half-up rounding (default one core). d2d_port_count counts
    manufactured ports, including inactive ones.
    D2D areas count PHY only. Extra hardware is a per-die module area before
    residual overhead. Custom dies retain one mean reference PCIe x8 strip by
    default; pcie_x8_count=None retains all strips for reference reconstruction.
    """
    _number("target_sm_count", target_sm_count, positive=True, integer=True)
    _number("target_l2_size_mb", target_l2_size_mb, positive=True)
    active = target_sm_count if active_sm_count is None else active_sm_count
    _number("active_sm_count", active, positive=True, integer=True)
    if active > target_sm_count:
        raise ValueError("Active SM count cannot exceed physical SM count")
    if not isinstance(include_display, bool):
        raise ValueError("include_display must be a boolean")
    data = load_soc_data(config_path, hw=hw)
    ref = data["reference"]
    cores, quantum, requested = _cpu_configuration(
        data, active, cpu_policy, cpu_core_count, cpu_core_quantum
    )
    _number("memory_bus_bits", memory_bus_bits, positive=True, integer=True)
    channel_bits = data["memory"]["channel_bits"]
    if memory_bus_bits % channel_bits:
        raise ValueError(f"memory_bus_bits must be a multiple of {channel_bits}")
    _validate_link_area(d2d_port_count, d2d_area_per_port_mm2)
    _validate_extra_phy_inputs(
        integration, d2d_port_count, d2d_phy_length_per_port_mm, extra_hardware_area_mm2
    )
    calibration = calibrate_overhead(config_path, hw=hw)
    alpha = calibration["overhead_ratio"] if overhead_ratio is None else overhead_ratio
    _number("overhead_ratio", alpha)
    components = dict(calibration["reference_components_mm2"])
    reference_pcie_count = len(
        data["measurement"]["components"]["pcie_phy"]["rectangles_px"]
    )
    pcie_count = reference_pcie_count if pcie_x8_count is None else pcie_x8_count
    _number("pcie_x8_count", pcie_count, integer=True)
    components["pcie_phy"] *= pcie_count / reference_pcie_count
    components["extra_hardware"] = extra_hardware_area_mm2
    gpu = calculate_gpu_area(
        ref["hardware"],
        target_sm_count,
        target_l2_size_mb,
        target_l1_size_kb,
        target_reg_size_kb,
    )
    components["gpu_compute_complex"] = gpu["area_breakdown_mm2"]["total"]
    for name in ("cpu_cores_and_private_cache", "cpu_shared_l3"):
        components[name] *= cores / ref["cpu_core_count"]
    components["memory_phy"] *= memory_bus_bits / ref["memory_bus_bits"]
    retained_dla = _apply_optional_components(
        components, data, include_display, include_dla
    )
    retained_vision, vision_module = _apply_vision_accelerator(
        components, data, include_vision_accelerator
    )
    components["d2d_interfaces"] = d2d_port_count * (d2d_area_per_port_mm2 or 0.0)
    warnings = [
        "First-order area estimate; CPU sufficiency and physical layout are not validated.",
        "Non-GPU rectangles are approximate manual measurements; residual includes unidentified system logic.",
        "CPU core quantum and cache scaling are design assumptions, not verified physical cluster topology.",
        "Memory PHY scales by width at the reference LPDDR technology; rate changes are not modeled.",
    ]
    if memory_controller_area_mm2 is None:
        components["memory_controller_explicit"] = 0.0
        warnings.append(
            "Memory controller is not separately identified; it is represented in residual overhead."
        )
    else:
        _number("memory_controller_area_mm2", memory_controller_area_mm2)
        if overhead_ratio is None:
            raise ValueError(
                "An explicit memory controller requires a recalibrated overhead_ratio to avoid double counting"
            )
        components["memory_controller_explicit"] = memory_controller_area_mm2
        warnings.append(
            "Caller must exclude explicit controller area from the residual calibration."
        )
    if not d2d_port_count:
        warnings.append(
            "No added inter-SoC link area: standalone die scenario, not a complete MicroPod ring die."
        )
    else:
        warnings.append(
            "D2D area counts PHY only; interconnect controller/adapter/buffer area is excluded by assumption."
        )
    warnings.extend(
        s
        for s in data.get("scope", [])
        if not s.startswith(
            ("New D2D/C2D", "New D2D/C2C", "Only an aggregate two-edge")
        )
    )
    warnings.append(
        "Custom PCIe x8 area/length uses the mean annotated reference x8 strip; reference calibration retains all strips."
    )
    warnings.append(
        "Edge-budget check includes memory, D2D PHY, USB, PCIe and HDMI/DP only; CSI/Ethernet remain in area but are outside this specified edge budget. Not a floorplan validation."
    )
    subtotal = sum(components.values())
    overhead = subtotal * alpha
    phy_length, length_source = _memory_phy_length(
        data, memory_bus_bits, memory_phy_total_length_mm
    )
    shoreline = check_memory_phy_shoreline(
        subtotal + overhead, phy_length, integration, phy_edge_count
    )
    lengths = {
        "memory_phy": phy_length,
        "d2d_phy": d2d_port_count * (d2d_phy_length_per_port_mm or 0.0),
        "usb_phy": _horizontal_phy_length(data, "usb_phy"),
        "pcie_phy": _horizontal_phy_length(data, "pcie_phy")
        * pcie_count
        / reference_pcie_count,
        "hdmi_dp_phy": _horizontal_phy_length(data, "display_phy")
        if include_display
        else 0.0,
    }
    io_shoreline = check_memory_phy_shoreline(
        subtotal + overhead, sum(lengths.values()), integration, phy_edge_count
    )
    io_shoreline["components_length_mm"] = lengths
    io_shoreline["d2d_length_known"] = (
        not d2d_port_count or d2d_phy_length_per_port_mm is not None
    )
    if io_shoreline["passed"] is False:
        io_shoreline["reason"] = (
            f"Combined memory + D2D + USB + PCIe + HDMI/DP PHY length exceeds {phy_edge_count} square-die edges"
        )
    return {
        "valid": io_shoreline["passed"] is not False,
        "constraints": {
            "memory_phy_shoreline": shoreline,
            "io_phy_shoreline": io_shoreline,
        },
        "model": f"{ref['hardware'].lower()}_soc_v2",
        "config": {
            "hardware": ref["hardware"],
            "physical_sm_count": target_sm_count,
            "active_sm_count": active,
            "l2_size_mb": target_l2_size_mb,
            "cpu_policy": cpu_policy,
            "cpu_policy_requested_cores": requested,
            "cpu_core_count": cores,
            "cpu_core_count_overridden": cpu_core_count is not None,
            "cpu_core_quantum": quantum,
            "cpu_cache_area_scale": cores / ref["cpu_core_count"],
            "memory_bus_bits": memory_bus_bits,
            "integration": integration,
            "phy_edge_count": phy_edge_count,
            "memory_phy_total_length_mm": phy_length,
            "memory_phy_length_source": length_source,
            "include_display": include_display,
            "include_dla": retained_dla,
            "include_vision_accelerator": retained_vision,
            "vision_accelerator_component": vision_module,
            "d2d_physical_port_count": d2d_port_count,
            "d2d_area_per_port_mm2": d2d_area_per_port_mm2,
            "d2d_phy_length_per_port_mm": d2d_phy_length_per_port_mm,
            "extra_hardware_area_mm2": extra_hardware_area_mm2,
            "pcie_x8_count": pcie_count,
            "overhead_ratio": alpha,
        },
        "area_breakdown_mm2": {
            "components": components,
            "modeled_subtotal": subtotal,
            "unidentified_overhead": overhead,
            "total": subtotal + overhead,
        },
        "gpu": gpu,
        "calibration": calibration,
        "warnings": warnings,
    }


def calculate_reference_soc_area(config_path=None, hw=None):
    """Reconstruct the full physical reference including all identified optional IP."""
    data = load_soc_data(config_path, hw=hw)
    ref = data["reference"]
    return calculate_soc_area(
        ref["physical_sm_count"],
        ref["l2_size_mb"],
        hw=ref["hardware"],
        active_sm_count=ref["active_sm_count"],
        cpu_policy="fixed",
        cpu_core_count=ref["cpu_core_count"],
        cpu_core_quantum=1,
        memory_bus_bits=ref["memory_bus_bits"],
        include_display=True,
        include_dla="dla" in data["measurement"]["components"],
        include_vision_accelerator=True,
        pcie_x8_count=None,
        config_path=config_path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hw", choices=("Thor", "Orin"), help="Hardware calibration (default: Thor)"
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Full hardware reference accounting reconstruction; excludes custom design options",
    )
    parser.add_argument("--sm", type=int, default=8, help="Physical SMs (default: 8)")
    parser.add_argument("--active-sm", type=int)
    parser.add_argument(
        "--l2-mb", type=float, help="L2 MB (default: half reference capacity)"
    )
    parser.add_argument("--memory-bits", type=int, default=128)
    parser.add_argument(
        "--integration",
        choices=("board", "package"),
        default="board",
        help="Apply the configured edge budget to package PHY interfaces",
    )
    parser.add_argument(
        "--memory-phy-length-mm",
        type=float,
        help="Override total physical memory PHY shoreline length in mm",
    )
    parser.add_argument(
        "--cpu-policy", choices=(*CPU_POLICIES, "all"), default="balanced"
    )
    parser.add_argument("--cpu-cores", type=int)
    parser.add_argument(
        "--cpu-quantum",
        type=int,
        help="CPU core granularity (default 1); scaled policies round half up",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--include-display",
        dest="include_display",
        action="store_true",
        help="Retain display subsystem and PHY (default)",
    )
    display.add_argument(
        "--exclude-display",
        dest="include_display",
        action="store_false",
        help="Remove display subsystem and dedicated PHY",
    )
    parser.set_defaults(include_display=True)
    dla = parser.add_mutually_exclusive_group()
    dla.add_argument("--include-dla", dest="include_dla", action="store_true")
    dla.add_argument("--exclude-dla", dest="include_dla", action="store_false")
    parser.set_defaults(include_dla=None)
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument(
        "--include-vision-accelerator",
        action="store_true",
        dest="include_vision_accelerator",
        help="Retain Thor DSP / Orin PVA",
    )
    vision.add_argument(
        "--exclude-vision-accelerator",
        action="store_false",
        dest="include_vision_accelerator",
        help="Remove Thor DSP / Orin PVA (default)",
    )
    parser.set_defaults(include_vision_accelerator=None)
    parser.add_argument("--d2d-ports", type=int, default=0)
    parser.add_argument("--d2d-area-per-port", type=float)
    parser.add_argument("--d2d-phy-length-per-port-mm", type=float)
    parser.add_argument("--extra-hardware-area-mm2", type=float, default=0.0)
    parser.add_argument("--pcie-x8-count", type=int, default=1)
    parser.add_argument("--phy-edge-count", type=int, choices=(2, 4), default=2)
    parser.add_argument("--overhead-ratio", type=float)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        if args.reference:
            import sys

            if any(
                item.split("=")[0] not in {"--reference", "--config", "--hw"}
                for item in sys.argv[1:]
                if item.startswith("--")
            ):
                parser.error(
                    "--reference cannot be combined with custom design options"
                )
            result = calculate_reference_soc_area(args.config, hw=args.hw)
        else:
            if args.cpu_policy == "all" and args.cpu_cores is not None:
                parser.error("--cpu-policy all cannot be combined with --cpu-cores")
            data = load_soc_data(args.config, hw=args.hw)
            l2_mb = (
                args.l2_mb
                if args.l2_mb is not None
                else data["reference"]["l2_size_mb"] / 2
            )
            policies = CPU_POLICIES if args.cpu_policy == "all" else [args.cpu_policy]
            results = {
                policy: calculate_soc_area(
                    args.sm,
                    l2_mb,
                    hw=args.hw,
                    active_sm_count=args.active_sm,
                    cpu_policy=policy,
                    cpu_core_count=args.cpu_cores,
                    cpu_core_quantum=args.cpu_quantum,
                    memory_bus_bits=args.memory_bits,
                    integration=args.integration,
                    phy_edge_count=args.phy_edge_count,
                    memory_phy_total_length_mm=args.memory_phy_length_mm,
                    include_display=args.include_display,
                    include_dla=args.include_dla,
                    include_vision_accelerator=args.include_vision_accelerator,
                    d2d_port_count=args.d2d_ports,
                    d2d_area_per_port_mm2=args.d2d_area_per_port,
                    d2d_phy_length_per_port_mm=args.d2d_phy_length_per_port_mm,
                    extra_hardware_area_mm2=args.extra_hardware_area_mm2,
                    pcie_x8_count=args.pcie_x8_count,
                    overhead_ratio=args.overhead_ratio,
                    config_path=args.config,
                )
                for policy in policies
            }
            result = results if args.cpu_policy == "all" else results[args.cpu_policy]
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
