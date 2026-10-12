"""One physical MicroPod die, organic-substrate S/M/L packages, USD costs."""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTUARY = ROOT / "cost_model" / "chiplet-actuary"
if not (ACTUARY / "chiplet_actuary").is_dir():
    raise ImportError(f"Chiplet Actuary checkout missing at {ACTUARY}")
sys.path.insert(0, str(ACTUARY))

# Imports follow explicit discovery of the vendored, hyphenated checkout.
from chiplet_actuary import spec  # noqa: E402
from chiplet_actuary.chip import Chip  # noqa: E402
from chiplet_actuary.module import Module  # noqa: E402
from chiplet_actuary.package import OS  # noqa: E402

from area_model.soc_model import calculate_soc_area  # noqa: E402
from icnt_model.icnt_model import calculate_ucie_phy_area  # noqa: E402


def number(name, value, *, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    if integer and value != int(value):
        raise ValueError(f"{name} must be an integer")
    return value


def select_ucie(target_bandwidth_gbps_per_port, config_path=None):
    """Raw one-direction bandwidth, smallest PHY area that meets the target."""
    target = number(
        "target_bandwidth_gbps_per_port", target_bandwidth_gbps_per_port, positive=True
    )
    path = Path(config_path) if config_path else ROOT / "icnt_model/configs/UCIE.json"
    raw = path.read_bytes()
    database = json.loads(raw)
    package = database["packages"]["standard"]
    choices = []
    for modules in database["available_module_count"]:
        number("module_count", modules, positive=True, integer=True)
        for lanes in package["available_lane_count"]:
            if lanes != 16:
                raise ValueError("Standard-package calibration requires x16 modules")
            for setting in package["configurations"]:
                rate = number("rate_gt", setting["rate_gt"], positive=True)
                bandwidth = modules * lanes * rate
                if bandwidth < target:
                    continue
                area = calculate_ucie_phy_area(
                    "standard", modules, lanes, rate, database
                )
                number("phy_area_mm2", area, positive=True)
                length = (
                    number("phy_width_um", package["phy_width_um"], positive=True)
                    * modules
                    / 1000
                )
                choices.append(
                    {
                        "package_type": "standard",
                        "modules_per_port": modules,
                        "lanes_per_module": lanes,
                        "rate_gt": rate,
                        "raw_bandwidth_gbps_per_port": bandwidth,
                        "raw_bandwidth_GBps_per_port": bandwidth / 8,
                        "phy_area_mm2_per_port": area,
                        "phy_length_mm_per_port": length,
                    }
                )
    if not choices:
        raise ValueError(
            "Requested bandwidth exceeds available standard UCIe configurations"
        )
    selected = min(
        choices,
        key=lambda c: (
            c["phy_area_mm2_per_port"],
            c["raw_bandwidth_gbps_per_port"],
            c["modules_per_port"],
            c["rate_gt"],
        ),
    )
    return dict(
        selected,
        target_bandwidth_gbps_per_port=target,
        physical_ports_per_die=2,
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        area_scope="PHY only; controller/adapter/buffer excluded",
        bandwidth_scope="raw single-direction per port; no efficiency factor",
    )


def _validate_tiers(tiers):
    if not isinstance(tiers, list) or len(tiers) != 3:
        raise ValueError("Require exactly S/M/L tiers with 1/2/4 dies")
    expected = {"S": 1, "M": 2, "L": 4}
    if {t["name"] for t in tiers} != set(expected):
        raise ValueError("Require unique S/M/L tier names")
    for tier in tiers:
        if set(tier) != {"name", "soc_count", "demand"}:
            raise ValueError("Each tier requires only name, soc_count, demand")
        n = number("soc_count", tier["soc_count"], positive=True, integer=True)
        if n != expected[tier["name"]]:
            raise ValueError("S/M/L require 1/2/4 identical physical dies")
        number("demand", tier["demand"], integer=True)
    total_demand = sum(t["demand"] for t in tiers)
    if total_demand <= 0:
        raise ValueError("At least one product tier must have demand")
    return total_demand


def _validate_config(config):
    unknown = set(config) - {
        "name",
        "node",
        "soc",
        "interconnect",
        "tiers",
        "whole_wafers",
    }
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    name = config.get("name", "micropod")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    node = str(config["node"])
    if node not in spec.Cost_Wafer_Die:
        raise ValueError("node must be an upstream Chiplet Actuary option")
    whole = config.get("whole_wafers", True)
    if not isinstance(whole, bool):
        raise ValueError("whole_wafers must be boolean")
    return name, node, whole, _validate_tiers(config["tiers"])


def _prepare_area(config, config_dir):
    interconnect = dict(config["interconnect"])
    if set(interconnect) - {"target_bandwidth_gbps_per_port", "ucie_config_path"}:
        raise ValueError(
            "Unknown interconnect option (efficiency factors are not supported)"
        )
    path = interconnect.get("ucie_config_path")
    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = Path(config_dir or Path.cwd()) / path
    link = select_ucie(interconnect["target_bandwidth_gbps_per_port"], path)
    soc = dict(config["soc"])
    controlled = {
        "integration",
        "phy_edge_count",
        "d2d_port_count",
        "d2d_area_per_port_mm2",
        "d2d_phy_length_per_port_mm",
        "pcie_x8_count",
    }
    if controlled & set(soc):
        raise ValueError(f"MicroPod derives these SoC options: {sorted(controlled)}")
    if "config_path" in soc:
        p = Path(soc["config_path"])
        soc["config_path"] = (
            p if p.is_absolute() else Path(config_dir or Path.cwd()) / p
        )
    area = calculate_soc_area(
        **soc,
        integration="package",
        phy_edge_count=2,
        d2d_port_count=2,
        pcie_x8_count=1,
        d2d_area_per_port_mm2=link["phy_area_mm2_per_port"],
        d2d_phy_length_per_port_mm=link["phy_length_mm_per_port"],
    )
    return link, area


def estimate_micropod(config, *, config_dir=None):
    name, node, whole, total_demand = _validate_config(config)
    link, area = _prepare_area(config, config_dir)
    tiers = config["tiers"]
    result = {
        "model": "micropod_mcm_v1",
        "name": name,
        "node": node,
        "currency": "USD",
        "feasible": area["valid"],
        "interconnect": link,
        "soc_area_result": area,
    }
    if not area["valid"]:
        result["reason"] = area["constraints"]["io_phy_shoreline"]["reason"]
        return result
    module = Module(name + "_design", node, area["area_breakdown_mm2"]["total"])
    chip = Chip(name + "_die", node, {module: 1})
    gross = chip.N_die_total()
    if gross <= 0 or not math.isfinite(gross):
        raise ValueError("Die exceeds upstream wafer geometry model")
    if not 0 < spec.bonding_yield_os <= 1:
        raise ValueError("Invalid upstream OS bonding yield")
    labels = (
        "raw_dies_and_bumps",
        "die_defect_loss",
        "raw_package",
        "package_defect_loss",
        "wasted_known_good_dies",
    )
    products = {}
    required_kgd = manufacturing = package_nre = 0.0
    for tier in sorted(tiers, key=lambda t: t["soc_count"]):
        label, n, demand = tier["name"], int(tier["soc_count"]), tier["demand"]
        package = OS(name + "_" + label, {chip: n})
        package_yield = spec.bonding_yield_os**n
        attempts = demand / package_yield
        kgd = n * attempts
        cost_parts = dict(zip(labels, package.cost_RE()))
        unit_cost = sum(cost_parts.values())
        # Only demanded products incur their independently designed package NRE.
        nre = package.NRE() if demand else 0.0
        edges = (
            []
            if n == 1
            else [
                {
                    "source_die": i,
                    "source_port": 1,
                    "destination_die": (i + 1) % n,
                    "destination_port": 0,
                }
                for i in range(n)
            ]
        )
        products[label] = {
            "soc_count": n,
            "demand": demand,
            "ring_links": edges,
            "physical_ports": 2 * n,
            "active_ports": 0 if n == 1 else 2 * n,
            "aggregate_resources": {
                "physical_sm_count": n * area["config"]["physical_sm_count"],
                "active_sm_count": n * area["config"]["active_sm_count"],
                "cpu_core_count": n * area["config"]["cpu_core_count"],
                "l2_mb_sum": n * area["config"]["l2_size_mb"],
                "memory_bus_bits_sum": n * area["config"]["memory_bus_bits"],
            },
            "package_type": "OS_single_die" if n == 1 else "OS_MCM",
            "package_area_mm2": package.area(),
            "package_yield": package_yield,
            "expected_package_attempts": attempts,
            "required_known_good_dies": kgd,
            "manufacturing_per_unit": unit_cost,
            "manufacturing_breakdown_per_unit": cost_parts,
            "manufacturing_total": unit_cost * demand,
            "package_nre": nre,
        }
        required_kgd += kgd
        manufacturing += unit_cost * demand
        package_nre += nre
    required_fabricated = required_kgd / chip.die_yield()
    fractional_wafers = required_fabricated / gross
    wafers = math.ceil(fractional_wafers) if whole else fractional_wafers
    rounding_cost = (wafers - fractional_wafers) * spec.Cost_Wafer_Die[node]
    manufacturing += rounding_cost
    nre = {
        "module_design_shared": module.NRE(),
        "die_design_shared": chip.NRE(),
        "package_designs": package_nre,
    }
    nre_total = sum(nre.values())
    result.update(
        {
            "tiers": products,
            "die": {
                "area_mm2": chip.area,
                "yield": chip.die_yield(),
                "gross_dies_per_wafer": gross,
                "known_good_die_cost": chip.cost_KGD(),
            },
            "production": {
                "whole_wafers": whole,
                "wafers": wafers,
                "required_known_good_dies": required_kgd,
                "fabricated_dies": wafers * gross,
                "unused_known_good_dies": max(
                    0.0, wafers * gross * chip.die_yield() - required_kgd
                ),
                "scrap_dies": wafers * gross * (1 - chip.die_yield()),
            },
            "family": {
                "shipped_systems": total_demand,
                "wafer_rounding_cost": rounding_cost,
                "manufacturing_total": manufacturing,
                "nre_breakdown": nre,
                "nre_total": nre_total,
                "total_cost": manufacturing + nre_total,
                "average_manufacturing_per_system": manufacturing / total_demand,
                "average_total_per_system": (manufacturing + nre_total) / total_demand,
            },
            "assumptions": [
                "One identical die design with two manufactured UCIe PHY ports in all products; no die harvesting.",
                "Separate appropriately sized organic-substrate package design per demanded SKU; common die/module NRE once.",
                "Upstream OS substrate multipliers model interconnection cost; no additional per-link charge.",
                "PHY dimensions come from the supplied UCIe database; no controller area or bandwidth efficiency factor.",
                "Extra hardware and PHY areas receive the SoC residual overhead factor once.",
                "Expected-value production, shared wafer pool; surplus receives no credit. Integer wafers do not guarantee delivery.",
                "Per-tier unit manufacturing uses fractional wafer costing; any family wafer-rounding cost is reported separately.",
                "Resource sums are not performance or shared-memory claims; timing, routing and memory packaging are not validated.",
                "External DRAM, PCB, power/cooling and software costs are excluded. No geometric process scaling.",
                "Upstream area-based NRE includes the complete custom die; no separately calibrated UCIe licensing charge.",
            ],
        }
    )
    return result
