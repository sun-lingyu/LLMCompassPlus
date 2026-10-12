import unittest

from area_model.soc_model import calculate_reference_soc_area, calculate_soc_area


class CpuRoundingTests(unittest.TestCase):
    def test_half_up_and_down_and_expansion(self):
        cases = [
            ("Thor", 6, "proportional", 4),  # 4.2 rounds down
            ("Thor", 7, "proportional", 5),  # 4.9 rounds up
            ("Thor", 15, "proportional", 11),  # 10.5 rounds up, not even
            ("Orin", 6, "proportional", 5),  # 4.5 rounds up, not even
            ("Orin", 8, "balanced", 9),
            ("Thor", 32, "proportional", 22),  # 22.4, formerly 24
            ("Thor", 32, "balanced", 18),  # 18.2, formerly 20
        ]
        for hw, sm, policy, cores in cases:
            with self.subTest(hw=hw, sm=sm, policy=policy):
                result = calculate_soc_area(sm, 4, hw=hw, cpu_policy=policy)
                self.assertEqual(result["config"]["cpu_core_count"], cores)
                self.assertEqual(result["config"]["cpu_core_quantum"], 1)
                ref_cores = 14 if hw == "Thor" else 12
                for module in ["cpu_shared_l3", "cpu_cores_and_private_cache"]:
                    self.assertAlmostEqual(
                        result["area_breakdown_mm2"]["components"][module],
                        result["calibration"]["reference_components_mm2"][module]
                        * cores
                        / ref_cores,
                    )

    def test_minimum_override_and_explicit_quantum(self):
        self.assertEqual(
            calculate_soc_area(1, 1, hw="Orin", cpu_policy="proportional")["config"][
                "cpu_core_count"
            ],
            1,
        )
        self.assertEqual(
            calculate_soc_area(8, 4, cpu_core_count=3)["config"]["cpu_core_count"], 3
        )
        # 4.2 rounds to 4 in units of two cores; old ceiling produced 6.
        self.assertEqual(
            calculate_soc_area(6, 4, cpu_policy="proportional", cpu_core_quantum=2)[
                "config"
            ]["cpu_core_count"],
            4,
        )

    def test_fixed_and_reference_preserved(self):
        for hw, cores, area in [("Thor", 14, 416), ("Orin", 12, 447.774)]:
            for sm in [1, 8, 32]:
                r = calculate_soc_area(sm, 4, hw=hw, cpu_policy="fixed")
                self.assertEqual(r["config"]["cpu_core_count"], cores)
                self.assertEqual(r["config"]["cpu_cache_area_scale"], 1)
            self.assertAlmostEqual(
                calculate_reference_soc_area(hw=hw)["area_breakdown_mm2"]["total"], area
            )
