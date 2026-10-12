import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from area_model.soc_model import calculate_reference_soc_area, calculate_soc_area
from cost_model.micropod import ROOT, estimate_micropod, select_ucie, spec


class MicroPodTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "cost_model/examples/micropod_5nm.json").read_text()
        )

    def test_ucie_selection_area_bandwidth_and_length(self):
        selected = select_ucie(1024)
        self.assertEqual(selected["modules_per_port"], 2)
        self.assertEqual(selected["rate_gt"], 32)
        self.assertEqual(selected["raw_bandwidth_GBps_per_port"], 128)
        self.assertAlmostEqual(selected["phy_area_mm2_per_port"], 1.76022)
        self.assertAlmostEqual(selected["phy_length_mm_per_port"], 1.143)
        self.assertGreaterEqual(select_ucie(1025)["raw_bandwidth_gbps_per_port"], 1025)
        for target in (0, -1, True, math.nan, 2049):
            with self.assertRaises(ValueError):
                select_ucie(target)

    def test_reference_closure_and_single_pcie(self):
        for hw in ("Thor", "Orin"):
            ref = calculate_reference_soc_area(hw=hw)
            self.assertAlmostEqual(
                ref["area_breakdown_mm2"]["total"],
                ref["calibration"]["reference_die_mm2"],
            )
            base = calculate_soc_area(8, 16, hw=hw)
            parts = base["area_breakdown_mm2"]["components"]
            count = ref["config"]["pcie_x8_count"]
            self.assertAlmostEqual(
                parts["pcie_phy"] * count,
                ref["calibration"]["reference_components_mm2"]["pcie_phy"],
            )

    def test_extra_area_overhead_once_and_calibration_unchanged(self):
        a = calculate_soc_area(8, 16)
        b = calculate_soc_area(8, 16, extra_hardware_area_mm2=10)
        self.assertEqual(a["calibration"], b["calibration"])
        self.assertAlmostEqual(
            b["area_breakdown_mm2"]["total"] - a["area_breakdown_mm2"]["total"],
            10 * (1 + a["config"]["overhead_ratio"]),
        )
        for value in (-1, math.inf, True):
            with self.assertRaises(ValueError):
                calculate_soc_area(8, 16, extra_hardware_area_mm2=value)

    def test_combined_constraint_not_memory_only(self):
        base = calculate_soc_area(
            8, 16, memory_phy_total_length_mm=29, integration="package"
        )
        self.assertTrue(base["constraints"]["memory_phy_shoreline"]["passed"])
        self.assertFalse(base["valid"])
        with self.assertRaises(ValueError):
            calculate_soc_area(
                8, 16, integration="package", d2d_port_count=2, d2d_area_per_port_mm2=1
            )

    def test_example_edge_sum_and_ring_ports(self):
        r = estimate_micropod(self.config)
        self.assertTrue(r["feasible"])
        constraint = r["soc_area_result"]["constraints"]["io_phy_shoreline"]
        expected = 2 * 283 / 392 * 16 / 2 + 2 * 1.143 + (47 + 211 / 2 + 64) * 26 / 645
        self.assertAlmostEqual(constraint["required_length_mm"], expected)
        for name, n in (("S", 1), ("M", 2), ("L", 4)):
            t = r["tiers"][name]
            self.assertEqual(t["physical_ports"], 2 * n)
            self.assertEqual(len(t["ring_links"]), 0 if n == 1 else n)
            used = [(e["source_die"], e["source_port"]) for e in t["ring_links"]]
            used += [
                (e["destination_die"], e["destination_port"]) for e in t["ring_links"]
            ]
            self.assertEqual(len(used), len(set(used)))

    def test_cost_matches_independent_os_formula_and_shared_nre(self):
        self.config["whole_wafers"] = False
        r = estimate_micropod(self.config)
        area = r["die"]["area_mm2"]
        kgd = r["die"]["known_good_die_cost"]
        for n, t in zip((1, 2, 4), r["tiers"].values()):
            pkgarea = area * n * spec.os_area_scale_factor
            factor = (
                1
                if n == 1
                else (2 if pkgarea > 900 else (1.75 if pkgarea > 289 else 1.5))
            )
            rawpkg = pkgarea * spec.cost_factor_os * factor
            expected = (
                n * (kgd + area * spec.c4_bump_cost_factor) + rawpkg
            ) / spec.bonding_yield_os**n
            self.assertAlmostEqual(t["manufacturing_per_unit"], expected)
        family = r["family"]
        self.assertAlmostEqual(
            family["manufacturing_total"],
            sum(t["manufacturing_total"] for t in r["tiers"].values()),
        )
        self.assertAlmostEqual(
            family["nre_breakdown"]["module_design_shared"],
            area * spec.Module_NRE_Cost_Factor["5"],
        )
        self.assertAlmostEqual(
            family["nre_breakdown"]["die_design_shared"],
            area * spec.Chip_NRE_Cost_Factor["5"] + spec.Chip_NRE_Cost_Fixed["5"],
        )
        self.assertAlmostEqual(
            family["total_cost"], family["manufacturing_total"] + family["nre_total"]
        )

    def test_shared_wafer_rounding_and_kgd_conservation(self):
        r = estimate_micropod(self.config)
        p, d = r["production"], r["die"]
        self.assertEqual(
            p["wafers"],
            math.ceil(
                p["required_known_good_dies"] / d["yield"] / d["gross_dies_per_wafer"]
            ),
        )
        self.assertAlmostEqual(
            p["fabricated_dies"],
            p["required_known_good_dies"]
            + p["unused_known_good_dies"]
            + p["scrap_dies"],
        )
        self.assertAlmostEqual(
            r["family"]["manufacturing_total"],
            sum(t["manufacturing_total"] for t in r["tiers"].values())
            + r["family"]["wafer_rounding_cost"],
        )

    def test_zero_demand_and_scaling_nre(self):
        a = estimate_micropod(self.config)
        for t in self.config["tiers"]:
            t["demand"] *= 2
        b = estimate_micropod(self.config)
        self.assertEqual(a["family"]["nre_total"], b["family"]["nre_total"])
        self.config["tiers"][0]["demand"] = 0
        self.assertEqual(estimate_micropod(self.config)["tiers"]["S"]["package_nre"], 0)

    def test_invalid_layout_has_no_cost(self):
        self.config["soc"]["memory_bus_bits"] = 512
        r = estimate_micropod(self.config)
        self.assertFalse(r["feasible"])
        self.assertNotIn("family", r)

    def test_bad_config_rejected(self):
        variants = []
        for key, value in (
            ("bandwidth_efficiency", 0.9),
            ("target_bandwidth_gbps_per_port", 2049),
        ):
            c = copy.deepcopy(self.config)
            c["interconnect"][key] = value
            variants.append(c)
        c = copy.deepcopy(self.config)
        c["soc"]["integration"] = "board"
        variants.append(c)
        c = copy.deepcopy(self.config)
        c["tiers"][1]["soc_count"] = 3
        variants.append(c)
        c = copy.deepcopy(self.config)
        c["whole_wafers"] = "false"
        variants.append(c)
        for c in variants:
            with self.assertRaises(ValueError):
                estimate_micropod(c)

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "cost_model.micropod_cli",
                    "--config",
                    str(ROOT / "cost_model/examples/micropod_5nm.json"),
                    "--output",
                    str(path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(json.loads(path.read_text())["feasible"])


if __name__ == "__main__":
    unittest.main()
