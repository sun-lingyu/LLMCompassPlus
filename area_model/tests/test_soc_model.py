import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from area_model.area_model import calculate_gpu_area
from area_model.soc_model import (
    calculate_reference_soc_area,
    calculate_soc_area,
    calibrate_overhead,
    check_memory_phy_shoreline,
    load_soc_data,
)


class SocAreaTests(unittest.TestCase):
    def test_legacy_gpu_formula(self):
        # Independent closed-form expectations from the existing calibration.
        # Command front uses the latest stepped region, excluding ISP.
        front = (632 * 66 + 401 * 23 + 504 * 108 + 126 * 27) / (1614 * 985) * 416
        expected = front + 1.57 + 16 * 22.68 / 32 + 8 * (3.11 + 26.85 / 22)
        self.assertAlmostEqual(
            calculate_gpu_area("Thor", 8, 16)["area_breakdown_mm2"]["total"], expected
        )
        self.assertAlmostEqual(
            calculate_gpu_area("Orin", 16, 4)["area_breakdown_mm2"]["total"], 121.27
        )

    def test_reference_accounting_closure(self):
        result = calculate_reference_soc_area()
        self.assertAlmostEqual(result["area_breakdown_mm2"]["total"], 416)
        self.assertEqual(result["config"]["physical_sm_count"], 22)
        self.assertEqual(result["config"]["active_sm_count"], 20)
        self.assertEqual(result["config"]["cpu_core_count"], 14)

    def test_cpu_policies_and_cache(self):
        for policy, expected in [("proportional", 6), ("balanced", 10), ("fixed", 14)]:
            result = calculate_soc_area(8, 16, cpu_policy=policy)
            self.assertEqual(result["config"]["cpu_core_count"], expected)
            self.assertAlmostEqual(
                result["config"]["cpu_cache_area_scale"], expected / 14
            )

    def test_disabled_sms_still_cost_area(self):
        first = calculate_soc_area(22, 32, active_sm_count=20, cpu_policy="fixed")
        second = calculate_soc_area(22, 32, active_sm_count=10, cpu_policy="fixed")
        self.assertEqual(first["area_breakdown_mm2"], second["area_breakdown_mm2"])

    def test_display_removed_without_changing_calibration(self):
        full = calculate_soc_area(
            22,
            32,
            active_sm_count=20,
            cpu_policy="fixed",
            memory_bus_bits=256,
        )
        cut = calculate_soc_area(
            22,
            32,
            active_sm_count=20,
            cpu_policy="fixed",
            memory_bus_bits=256,
            include_display=False,
        )
        self.assertTrue(full["config"]["include_display"])
        comp = full["area_breakdown_mm2"]["components"]
        self.assertGreater(comp["display_subsystem"], 0)
        self.assertGreater(comp["display_phy"], 0)
        expected_delta = (comp["display_subsystem"] + comp["display_phy"]) * (
            1 + full["config"]["overhead_ratio"]
        )
        self.assertAlmostEqual(
            full["area_breakdown_mm2"]["total"] - cut["area_breakdown_mm2"]["total"],
            expected_delta,
        )
        self.assertEqual(full["calibration"], cut["calibration"])
        self.assertGreater(cut["area_breakdown_mm2"]["components"]["video_codec"], 0)

    def test_memory_width_only_changes_phy_before_overhead(self):
        a = calculate_soc_area(8, 16, memory_bus_bits=128)
        b = calculate_soc_area(8, 16, memory_bus_bits=256)
        ac = a["area_breakdown_mm2"]["components"]
        bc = b["area_breakdown_mm2"]["components"]
        self.assertAlmostEqual(ac["memory_phy"] * 2, bc["memory_phy"])
        for name in ac:
            if name != "memory_phy":
                self.assertEqual(ac[name], bc[name])

    def test_added_links_require_explicit_area(self):
        with self.assertRaises(ValueError):
            calculate_soc_area(8, 16, d2d_port_count=2)
        base = calculate_soc_area(8, 16)
        # Synthetic unit-test input, not a physical link calibration.
        links = calculate_soc_area(8, 16, d2d_port_count=2, d2d_area_per_port_mm2=1.25)
        expected = 2.5 * (1 + base["config"]["overhead_ratio"])
        self.assertAlmostEqual(
            links["area_breakdown_mm2"]["total"] - base["area_breakdown_mm2"]["total"],
            expected,
        )

    def test_explicit_controller_requires_residual_recalibration(self):
        with self.assertRaises(ValueError):
            calculate_soc_area(8, 16, memory_controller_area_mm2=1)

    def test_override_and_minimum_cpu(self):
        self.assertEqual(
            calculate_soc_area(1, 1, cpu_policy="proportional")["config"][
                "cpu_core_count"
            ],
            1,
        )
        result = calculate_soc_area(8, 16, cpu_core_count=4)
        self.assertEqual(result["config"]["cpu_core_count"], 4)
        self.assertTrue(result["config"]["cpu_core_count_overridden"])

    def test_invalid_inputs(self):
        for kwargs in [
            {"active_sm_count": 9},
            {"cpu_policy": "typo"},
            {"cpu_core_count": 3, "cpu_core_quantum": 2},
            {"cpu_core_quantum": 0},
            {"memory_bus_bits": 129},
            {"overhead_ratio": -0.1},
            {"overhead_ratio": math.nan},
            {"d2d_port_count": -1},
            {"d2d_port_count": 2, "d2d_area_per_port_mm2": 0},
            {"d2d_area_per_port_mm2": 2},
            {"include_display": "false"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calculate_soc_area(8, 16, **kwargs)
        for sm, l2 in [(0, 16), (8, 0), (2.5, 16), (8, math.inf), (True, 16)]:
            with self.subTest(sm=sm, l2=l2), self.assertRaises(ValueError):
                calculate_soc_area(sm, l2)

    def test_no_implicit_working_directory_or_stdout(self):
        cwd = os.getcwd()
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()) as capture,
        ):
            try:
                os.chdir(tmp)
                calculate_soc_area(8, 16)
            finally:
                os.chdir(cwd)
        self.assertEqual(capture.getvalue(), "")

    def test_rectangles_cannot_overlap(self):
        data = load_soc_data()
        parts = data["measurement"]["components"]
        parts["isp"]["rectangles_px"] = parts["dsp"]["rectangles_px"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                calibrate_overhead(path)

    def test_cli_machine_readable_and_policy_comparison(self):
        root = Path(__file__).resolve().parents[2]
        process = subprocess.run(
            [sys.executable, "-B", "-m", "area_model.soc_model", "--cpu-policy", "all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(process.stdout)
        self.assertTrue(all(r["config"]["include_display"] for r in result.values()))
        self.assertEqual(set(result), {"fixed", "balanced", "proportional"})
        totals = [
            result[p]["area_breakdown_mm2"]["total"]
            for p in ("proportional", "balanced", "fixed")
        ]
        self.assertEqual(totals, sorted(totals))

    def test_package_shoreline_threshold(self):
        self.assertTrue(check_memory_phy_shoreline(256, 32, "package")["passed"])
        self.assertTrue(check_memory_phy_shoreline(257, 32, "package")["passed"])
        self.assertFalse(check_memory_phy_shoreline(255, 32, "package")["passed"])
        self.assertIsNone(check_memory_phy_shoreline(255, 32, "board")["passed"])

    def test_package_invalid_does_not_inflate_area(self):
        board = calculate_soc_area(8, 16, memory_phy_total_length_mm=100)
        package = calculate_soc_area(
            8, 16, integration="package", memory_phy_total_length_mm=100
        )
        self.assertTrue(board["valid"])
        self.assertFalse(package["valid"])
        self.assertEqual(board["area_breakdown_mm2"], package["area_breakdown_mm2"])
        self.assertIsNotNone(package["constraints"]["memory_phy_shoreline"]["reason"])

    def test_phy_length_uses_vertical_scale_and_bus_width(self):
        base = calculate_soc_area(8, 16, integration="package", memory_bus_bits=256)
        half = calculate_soc_area(8, 16, integration="package", memory_bus_bits=128)
        expected = 2 * 283 / 392 * 16
        self.assertAlmostEqual(base["config"]["memory_phy_total_length_mm"], expected)
        self.assertAlmostEqual(
            half["config"]["memory_phy_total_length_mm"], expected / 2
        )
        self.assertTrue(half["valid"])

    def test_invalid_shoreline_inputs(self):
        for kwargs in [
            {"integration": "typo"},
            {"memory_phy_total_length_mm": 0},
            {"memory_phy_total_length_mm": -1},
            {"memory_phy_total_length_mm": math.nan},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calculate_soc_area(8, 16, **kwargs)

    def test_cli_package_invalid_flag(self):
        root = Path(__file__).resolve().parents[2]
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "area_model.soc_model",
                "--integration",
                "package",
                "--memory-phy-length-mm",
                "100",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertFalse(json.loads(process.stdout)["valid"])

    def test_no_silent_reference_override(self):
        root = Path(__file__).resolve().parents[2]
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "area_model.soc_model",
                "--reference",
                "--sm",
                "8",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
