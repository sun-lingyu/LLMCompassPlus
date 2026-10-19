import unittest
from math import ceil
from types import SimpleNamespace

from software_model.layernorm import FusedLayerNorm, L2CacheLayerNorm
from software_model.utils import L2AccessType, Tensor, data_type_dict as dt


class LayerNormModelTests(unittest.TestCase):
    def setUp(self):
        self.device = SimpleNamespace(
            compute_module=SimpleNamespace(
                l2_size=1024 * 1024,
                clock_freq=1e9,
                l2_bandwidth_per_cycle=128,
                compute_latency_factor={"layernorm": 0},
            ),
            io_module=SimpleNamespace(
                bandwidth=32e9, bandwidth_efficiency=0.8,
            ),
        )

    def test_residual_dtype_required(self):
        with self.assertRaises(TypeError):
            FusedLayerNorm(dt["fp16"], dt["fp8"])
        with self.assertRaises(AssertionError):
            FusedLayerNorm(dt["fp16"], dt["fp8"], None)
        with self.assertRaises(AssertionError):
            FusedLayerNorm(dt["fp16"], dt["fp8"], dt["fp8"])
        with self.assertRaises(TypeError):
            L2CacheLayerNorm(1024, 16, 16, dt["fp16"], dt["fp8"])
        with self.assertRaises(AssertionError):
            L2CacheLayerNorm(1024, 16, 16, dt["fp16"], dt["fp8"], None)

    def test_alignment_and_output_dtypes(self):
        for shape in ([1, 17], [17, 33], [16, 32], [2, 3, 17]):
            for output_dtype in (dt["fp16"], dt["fp8"]):
                with self.subTest(shape=shape, output_dtype=output_dtype.name):
                    op = FusedLayerNorm(dt["fp16"], output_dtype, dt["fp16"])
                    output, residual = op(
                        Tensor(list(shape), dt["fp16"]),
                        Tensor(list(shape), dt["fp16"]),
                    )
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(residual.shape, shape)
                    self.assertIs(output.dtype, output_dtype)
                    self.assertIs(residual.dtype, dt["fp16"])
                    bytes_per_element = 6 + output_dtype.word_size
                    self.assertEqual(op.io_size, op.M * op.N * bytes_per_element)
                    latency = op.compile_and_simulate(self.device, drain_at_end=True)
                    expected = ceil(op.M / 16) * ceil(op.N / 16) * 256 * bytes_per_element
                    self.assertEqual(op.mem_access_size, expected)
                    self.assertEqual(op.l2_status.occupied_size, expected)
                    expected_cycles = op.execution_cycles + op.drain_cycles
                    self.assertEqual(latency, expected_cycles / 1e9)
                    self.assertEqual(op.roofline_model(self.device), op.io_size / 32e9)

    def test_previous_cache_and_eviction(self):
        previous = L2CacheLayerNorm(
            1024 * 1024, 1, 17, dt["fp16"], dt["fp16"], dt["fp16"]
        )
        previous.access(L2AccessType.OUTPUT, (0, 0), (16, 32))
        op = FusedLayerNorm(dt["fp16"], dt["fp8"], dt["fp16"])
        op(Tensor([1, 17], dt["fp16"]), Tensor([1, 17], dt["fp16"]))
        op.compile_and_simulate(self.device, previous)
        self.assertEqual(op.mem_access_size, 16 * 32 * 2)
        self.assertEqual(op.l2_status.pending_writeback_bytes, 16*32*5)
        self.assertEqual(previous.occupied_size, 16 * 32 * 2)
        self.assertTrue(all(t.access_type == L2AccessType.OUTPUT for t in previous.resident_tiles))
        self.device.compute_module.l2_size = 512
        op.compile_and_simulate(self.device, drain_at_end=True)
        self.assertEqual(op.mem_access_size, 16 * 32 * 7)
        self.assertLessEqual(op.l2_status.occupied_size, 512)


if __name__ == "__main__":
    unittest.main()
