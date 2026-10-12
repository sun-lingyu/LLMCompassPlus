import unittest

from area_model.soc_model import calculate_reference_soc_area, calculate_soc_area


class VisionRemovalTests(unittest.TestCase):
    def test_removal_preserves_reference_and_other_components(self):
        for hw, module, l2 in [("Thor", "dsp", 16), ("Orin", "pva", 2)]:
            with self.subTest(hw=hw):
                base = calculate_soc_area(8, l2, hw=hw, integration="package")
                kept = calculate_soc_area(8, l2, hw=hw, include_vision_accelerator=True)
                a = base["area_breakdown_mm2"]
                b = kept["area_breakdown_mm2"]
                self.assertEqual(a["components"][module], 0)
                self.assertFalse(base["config"]["include_vision_accelerator"])
                self.assertTrue(base["valid"])
                self.assertEqual(base["calibration"], kept["calibration"])
                self.assertAlmostEqual(
                    b["total"] - a["total"],
                    b["components"][module] * (1 + base["config"]["overhead_ratio"]),
                )
                for name in a["components"]:
                    if name != module:
                        self.assertEqual(a["components"][name], b["components"][name])
                ref = calculate_reference_soc_area(hw=hw)
                self.assertTrue(ref["config"]["include_vision_accelerator"])
                self.assertGreater(ref["area_breakdown_mm2"]["components"][module], 0)
                self.assertAlmostEqual(
                    ref["area_breakdown_mm2"]["total"],
                    ref["calibration"]["reference_die_mm2"],
                )

    def test_system_bundle_reclassified_without_losing_reference_area(self):
        ref = calculate_reference_soc_area(hw="Orin")["calibration"]
        pva = 627 * 394 * 447.774 / (2229 * 2014)
        old_bundle = (570 * 258 - 90 * 34) * 447.774 / (1117 * 1012)
        self.assertAlmostEqual(ref["reference_residual_mm2"], 94.482889111986)
        self.assertAlmostEqual(
            ref["reference_components_mm2"]["system_components"], old_bundle - pva
        )
        for name in ("pva_system", "security_safety", "dma"):
            self.assertNotIn(name, ref["reference_components_mm2"])

    def test_system_area_fixed_across_resource_scaling(self):
        for hw, sm, l2 in [("Thor", 20, 32), ("Orin", 16, 4)]:
            ref = calculate_reference_soc_area(hw=hw)
            expected = ref["area_breakdown_mm2"]["components"]["system_components"]
            self.assertGreater(expected, 0)
            for policy in ["fixed", "balanced", "proportional"]:
                for bits in [128, 256]:
                    result = calculate_soc_area(
                        8, l2 / 2, hw=hw, cpu_policy=policy, memory_bus_bits=bits
                    )
                    self.assertEqual(
                        result["area_breakdown_mm2"]["components"]["system_components"],
                        expected,
                    )
            full = calculate_soc_area(sm, l2, hw=hw)
            self.assertEqual(
                full["area_breakdown_mm2"]["components"]["system_components"], expected
            )

    def test_invalid_toggle(self):
        with self.assertRaises(ValueError):
            calculate_soc_area(8, 16, include_vision_accelerator="false")
