import copy
import json
import unittest
from pathlib import Path

from hardware_model.device import _create_device_from_config, device_dict
from software_model.elementwise_like import ElementwiseLike, L2CacheElementwiseLike
from software_model.layernorm import FusedLayerNorm
from software_model.rope import FusedQKNormRoPE
from software_model.silu import L2CacheSiLU, SiLU
from software_model.utils import L2AccessType, Tensor
from software_model.utils import data_type_dict as dt


class ElementwiseLikeTests(unittest.TestCase):
    def make_operator(self, name):
        if name == "layernorm":
            op = FusedLayerNorm(dt["fp16"], dt["fp8"], dt["fp16"])
            op(Tensor([1, 17], dt["fp16"]), Tensor([1, 17], dt["fp16"]))
            return op, 17
        if name == "rope":
            op = FusedQKNormRoPE(dt["fp16"])
            qkv = Tensor([1, 48], dt["fp16"])
            self.assertIs(op(qkv, 8, 2, 2), qkv)
            return op, 32
        op = SiLU(dt["fp16"], dt["fp8"], fused_mul=True)
        self.assertEqual(op(Tensor([1, 34], dt["fp16"])).shape, [1, 17])
        return op, 17

    def test_overlap_and_logical_output_size(self):
        for device_name in ("Orin", "Thor"):
            for name in ("layernorm", "rope", "silu"):
                with self.subTest(device=device_name, operator=name):
                    device = copy.deepcopy(device_dict[device_name])
                    op, expected_numel = self.make_operator(name)
                    self.assertIsInstance(op, ElementwiseLike)
                    self.assertEqual(op.output_numel, expected_numel)
                    device.compute_module.compute_latency_factor[name] = 0
                    baseline = op.compile_and_simulate(device)
                    roofline = op.roofline_model(device)
                    self.assertIsInstance(op.l2_status, L2CacheElementwiseLike)
                    self.assertEqual(op.compute_latency, 0)
                    self.assertEqual(baseline, op.memory_latency)
                    memory_bytes = op.mem_access_size
                    residency = list(op.l2_status.resident_tiles)
                    for multiplier in (0.25, 1, 4, 0):
                        device.compute_module.compute_latency_factor[name] = (
                            multiplier * baseline / expected_numel
                        )
                        latency = op.compile_and_simulate(device)
                        self.assertAlmostEqual(
                            op.compute_latency, multiplier * baseline
                        )
                        self.assertEqual(op.memory_latency, baseline)
                        self.assertAlmostEqual(
                            latency, max(baseline, multiplier * baseline)
                        )
                        self.assertEqual(op.mem_access_size, memory_bytes)
                        self.assertEqual(list(op.l2_status.resident_tiles), residency)
                        self.assertEqual(op.roofline_model(device), roofline)

    def test_plain_silu_and_dtype_independent_factor(self):
        device = copy.deepcopy(device_dict["Thor"])
        device.compute_module.compute_latency_factor["silu"] = 1e-6
        for dtype in (dt["fp16"], dt["fp8"], dt["fp4"]):
            op = SiLU(dtype)
            op(Tensor([2, 3, 17], dtype))
            op.compile_and_simulate(device)
            self.assertEqual(op.output_numel, 102)
            self.assertEqual(op.compute_latency, 102e-6)
            self.assertEqual(op.latency, op.compute_latency)

    def test_fp4_scale_cache_inheritance(self):
        device = copy.deepcopy(device_dict["Thor"])
        previous = L2CacheSiLU(
            device.compute_module.l2_size, 16, 64, dt["fp4"], dt["fp4"]
        )
        previous.access(L2AccessType.OUTPUT, (0, 0), (16, 64))
        previous.access(L2AccessType.OUTPUT_SCALE, (0, 0), (16, 16))
        previous_tiles = list(previous.resident_tiles)
        previous_bytes = previous.occupied_size
        op = SiLU(dt["fp4"], fused_mul=True)
        op(Tensor([16, 64], dt["fp4"]))
        op.compile_and_simulate(device, previous)
        self.assertEqual(op.mem_access_size, 0)  # inherited input hits; outputs remain dirty
        self.assertEqual(op.l2_status.pending_writeback_bytes, previous_bytes + 16 * 32 * 0.5 + 16 * 16)
        self.assertEqual(list(previous.resident_tiles), previous_tiles)
        self.assertEqual(previous.occupied_size, previous_bytes)
        types = {tile.access_type for tile in op.l2_status.resident_tiles}
        self.assertEqual(
            types,
            {
                L2AccessType.ACTIVATION,
                L2AccessType.ACTIVATION_SCALE,
                L2AccessType.OUTPUT,
                L2AccessType.OUTPUT_SCALE,
            },
        )

    def test_device_config_loading(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("Orin", "Thor"):
            config = json.loads(
                (root / "hardware_model" / "configs" / (name + ".json")).read_text()
            )
            self.assertEqual(
                _create_device_from_config(config).compute_module.compute_latency_factor,
                {"quant": 0.0, **config["compute_module"]["compute_latency_factor"]},
            )
            expected = {"layernorm": 1e-9, "rope": 2e-9, "silu": 3e-9, "quant": 0}
            config["compute_module"]["compute_latency_factor"] = expected.copy()
            device = _create_device_from_config(config)
            self.assertEqual(device.compute_module.compute_latency_factor, expected)
            config["compute_module"]["compute_latency_factor"]["rope"] = 0
            self.assertEqual(device.compute_module.compute_latency_factor, expected)
            del config["compute_module"]["compute_latency_factor"]
            self.assertEqual(
                _create_device_from_config(
                    config
                ).compute_module.compute_latency_factor,
                {"layernorm": 0, "rope": 0, "silu": 0, "quant": 0},
            )
            for bad in (-1, float("nan"), float("inf")):
                config["compute_module"]["compute_latency_factor"] = {"rope": bad}
                with self.assertRaises(ValueError):
                    _create_device_from_config(config)


if __name__ == "__main__":
    unittest.main()
