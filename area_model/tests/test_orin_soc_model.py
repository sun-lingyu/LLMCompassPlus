import json
import subprocess
import sys
import unittest
from pathlib import Path

from area_model.soc_model import (
    calculate_reference_soc_area,
    calculate_soc_area,
    calibrate_overhead,
    load_soc_data,
)


class OrinSocTests(unittest.TestCase):
    def test_full_reference_closes_with_dla(self):
        result = calculate_reference_soc_area(hw="Orin")
        self.assertAlmostEqual(result["area_breakdown_mm2"]["total"], 20.17 * 22.2)
        self.assertEqual(result["config"]["cpu_core_count"], 12)
        self.assertTrue(result["config"]["include_dla"])
        self.assertGreater(result["area_breakdown_mm2"]["components"]["dla"], 0)

    def test_custom_removes_dla_without_recalibration(self):
        common = {"hw": "Orin", "cpu_policy": "fixed", "memory_bus_bits": 256}
        full = calculate_soc_area(16, 4, include_dla=True, **common)
        custom = calculate_soc_area(16, 4, **common)
        a, b = full["area_breakdown_mm2"], custom["area_breakdown_mm2"]
        self.assertEqual(b["components"]["dla"], 0)
        self.assertEqual(full["calibration"], custom["calibration"])
        expected = a["components"]["dla"] * (1 + full["config"]["overhead_ratio"])
        self.assertAlmostEqual(a["total"] - b["total"], expected)
        for name in a["components"]:
            if name != "dla":
                self.assertEqual(a["components"][name], b["components"][name])

    def test_cpu_policies_pva_and_slc(self):
        for policy, cores in [("proportional", 6), ("balanced", 9), ("fixed", 12)]:
            result = calculate_soc_area(8, 2, hw="Orin", cpu_policy=policy)
            self.assertEqual(result["config"]["cpu_core_count"], cores)
            parts = result["area_breakdown_mm2"]["components"]
            self.assertEqual(parts["pva"], 0)
            reference = result["calibration"]["reference_components_mm2"]
            self.assertGreater(reference["pva"], 0)
            self.assertEqual(parts["system_slc"], reference["system_slc"])
            self.assertAlmostEqual(parts["gpu_compute_complex"], 71.55)

    def test_full_region_accounting_no_duplicate_children(self):
        reference = calibrate_overhead(hw="Orin")
        parts = reference["reference_components_mm2"]
        scale = 447.774 / (1117 * 1012)
        self.assertAlmostEqual(
            parts["gpu_compute_complex"] + parts["gpu_region_other_fixed"],
            (765 * 147 + 529 * 474) * scale,
        )
        self.assertAlmostEqual(parts["pva"], 627 * 394 * 447.774 / (2229 * 2014))
        self.assertAlmostEqual(
            parts["cpu_cores_and_private_cache"] + parts["cpu_shared_l3"],
            394 * 274 * scale,
        )
        for name in (
            "gpu_region",
            "pva_system",
            "security_safety",
            "dma",
            "multimedia_complex",
        ):
            self.assertNotIn(name, parts)

    def test_gpu_support_is_fixed_not_hidden_in_global_residual(self):
        a = calculate_soc_area(16, 4, hw="Orin")
        b = calculate_soc_area(8, 2, hw="Orin")
        self.assertEqual(
            a["area_breakdown_mm2"]["components"]["gpu_region_other_fixed"],
            b["area_breakdown_mm2"]["components"]["gpu_region_other_fixed"],
        )

    def test_mixed_phy_directions_and_rotated_dimensions(self):
        data = load_soc_data(hw="Orin")
        m = data["measurement"]
        x0, y0, x1, y1 = m["die_bounds_px"]
        boxes = m["components"]["memory_phy"]["rectangles_px"]
        expected = sum(r - left for left, _, r, _ in boxes[:2]) * 22.2 / (x1 - x0)
        expected += sum(b - top for _, top, _, b in boxes[2:]) * 20.17 / (y1 - y0)
        full = calculate_soc_area(8, 2, hw="Orin", memory_bus_bits=256)
        half = calculate_soc_area(8, 2, hw="Orin", memory_bus_bits=128)
        self.assertAlmostEqual(full["config"]["memory_phy_total_length_mm"], expected)
        self.assertAlmostEqual(
            half["config"]["memory_phy_total_length_mm"], expected / 2
        )

    def test_package_constraint_is_applied(self):
        result = calculate_soc_area(
            8, 2, hw="Orin", integration="package", memory_phy_total_length_mm=100
        )
        self.assertFalse(result["valid"])

    def test_selector_and_config_mismatch(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "Orin_soc.json"
        self.assertEqual(load_soc_data(path)["reference"]["hardware"], "Orin")
        self.assertAlmostEqual(calibrate_overhead(path)["reference_die_mm2"], 447.774)
        with self.assertRaises(ValueError):
            load_soc_data(path, hw="Thor")
        with self.assertRaises(ValueError):
            load_soc_data(hw="unknown")

    def test_unidentified_display_cannot_be_silently_removed(self):
        with self.assertRaises(ValueError):
            calculate_soc_area(8, 2, hw="Orin", include_display=False)
        with self.assertRaises(ValueError):
            calculate_soc_area(8, 16, hw="Thor", include_dla=True)

    def test_thor_reference_and_fixed_system_accounting(self):
        self.assertAlmostEqual(
            calculate_reference_soc_area()["area_breakdown_mm2"]["total"], 416
        )
        result = calculate_soc_area(
            8, 16, cpu_policy="balanced", include_vision_accelerator=True
        )
        self.assertAlmostEqual(
            result["calibration"]["reference_components_mm2"]["system_components"],
            483 * 62 * 416 / (1011 * 617),
        )

    def test_orin_cli_defaults_and_reference(self):
        cwd = Path(__file__).resolve().parents[2]
        base = [sys.executable, "-B", "-m", "area_model.soc_model", "--hw", "Orin"]
        result = json.loads(
            subprocess.run(
                base, cwd=cwd, capture_output=True, text=True, check=True
            ).stdout
        )
        self.assertEqual(result["config"]["l2_size_mb"], 2)
        self.assertFalse(result["config"]["include_dla"])
        result = json.loads(
            subprocess.run(
                base + ["--reference"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        self.assertTrue(result["config"]["include_dla"])


if __name__ == "__main__":
    unittest.main()
