"""Regression coverage for the RoPE/SiLU layer integration (unittest)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulate import main as sim
from hardware_model.device import device_dict
from software_model.layernorm import FusedLayerNorm
from software_model.rope import FusedQKNormRoPE, L2CacheQKNormRoPE
from software_model.silu import SiLU, L2CacheSiLU
from software_model.utils import L2AccessType, Tensor, data_type_dict as dt
from test.flashattn.utils import get_output_dtype as fa_dtype
from test.layernorm.utils import get_output_dtype as ln_dtype
from test.matmul.utils import get_output_dtype as mm_dtype
from test.rope.utils import get_output_dtype as rope_dtype
from test.silu.utils import get_output_dtype as silu_dtype


class IntegrationTests(unittest.TestCase):
    def test_dtype_contract(self):
        for is_test in (False, True):
            self.assertIs(fa_dtype(dt["fp8"], is_test), dt["fp16"])
            self.assertIs(rope_dtype(dt["fp8"], is_test), dt["fp16"])
            self.assertIs(ln_dtype(dt["fp8"], is_test), dt["fp8"])
            self.assertIs(silu_dtype(dt["fp8"], is_test), dt["fp8"])
            for op in ("qkv_proj", "o_proj", "up_proj", "down_proj"):
                self.assertIs(mm_dtype(dt["fp8"], op, is_test), dt["fp16"])
        for name in ("fp16", "int8", "int4", "fp4"):
            expected = "fp16" if name == "int4" else name
            for helper in (fa_dtype, ln_dtype, rope_dtype):
                self.assertIs(helper(dt[name], False), dt[expected])
            self.assertIs(fa_dtype(dt[name], True), dt["fp8" if name == "fp4" else "fp16"])
            self.assertIs(ln_dtype(dt[name], True), dt["fp16"])
            self.assertIs(rope_dtype(dt[name], True), dt["fp16"])

    def test_fused_silu_traffic_and_residency(self):
        pcb = device_dict["Thor"]
        op = SiLU(dt["fp16"], dt["fp8"], fused_mul=True)
        result = op(Tensor([32, 64], dt["fp16"]))
        self.assertEqual(result.shape, [32, 32])
        self.assertEqual(op.io_size, 32 * (64 * 2 + 32))
        op.compile_and_simulate(pcb, drain_at_end=True)
        self.assertEqual(op.mem_access_size, op.io_size)
        previous = L2CacheSiLU(pcb.compute_module.l2_size, 32, 64, dt["fp16"], dt["fp16"])
        previous.access(L2AccessType.OUTPUT, (0, 0), (32, 64))
        op.compile_and_simulate(pcb, previous)
        self.assertEqual(op.mem_access_size, 0)
        self.assertEqual(op.l2_status.pending_writeback_bytes, 32*64*2 + 32*32)
        self.assertEqual(previous.occupied_size, 32 * 64 * 2)
        odd = SiLU(dt["fp16"], fused_mul=True)
        with self.assertRaises(AssertionError):
            odd(Tensor([1, 33], dt["fp16"]))
        small = SiLU(dt["fp16"], dt["fp8"], fused_mul=True)
        self.assertEqual(small(Tensor([1, 34], dt["fp16"])).shape, [1, 17])
        small.compile_and_simulate(pcb, drain_at_end=True)
        self.assertEqual(small.mem_access_size, 16 * 48 * 2 + 16 * 32)

    def test_fp16_residual_traffic(self):
        op = FusedLayerNorm(dt["fp16"], dt["fp8"], residual_dtype=dt["fp16"])
        output, residual = op(Tensor([16, 32], dt["fp16"]), Tensor([16, 32], dt["fp16"]))
        self.assertIs(output.dtype, dt["fp8"])
        self.assertIs(residual.dtype, dt["fp16"])
        op.compile_and_simulate(device_dict["Thor"], drain_at_end=True)
        self.assertEqual(op.mem_access_size, 16 * 32 * 7)
        self.assertEqual(op.io_size, op.mem_access_size)
        self.assertEqual(op.l2_status.residual_tile_size, 512)

    def test_rope_keeps_resident_v(self):
        pcb = device_dict["Thor"]
        previous = L2CacheQKNormRoPE(pcb.compute_module.l2_size, 16, 96, dt["fp16"], dt["fp16"])
        previous.access(L2AccessType.OUTPUT, (0, 0), (16, 96))
        op = FusedQKNormRoPE(dt["fp16"])
        op(Tensor([16, 96], dt["fp16"]), 16, 2, 2)
        op.compile_and_simulate(pcb, previous)
        self.assertEqual(op.mem_access_size, 0)
        self.assertIn(op.l2_status.Tile(L2AccessType.OUTPUT, (0, 80)), op.l2_status.resident_tiles)

    def test_cache_migration(self):
        meta = dict(device="Thor", model="Qwen3_0_6B", precision="fp8", phase="prefill", spec_tokens=64, seq_len=64, is_causal=True, parallelism="TP", degree=1)
        row = dict(meta, **{op: dict(lat_ms=1, dram_access_byte=2, power_w=3) for op in sim.LAYER_COMPUTE_OP_NAMES})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            self.assertEqual(sim.load_layer_compute_cache(str(path)), {})
            old = dict(row)
            old.pop("rope")
            old.pop("silu")
            path.write_text(json.dumps([old]))
            self.assertEqual(sim.load_layer_compute_cache(str(path)), {})
            sim.update_layer_cache_record(row, str(path))
            sim.update_layer_cache_record(row, str(path))
            records = json.loads(path.read_text())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["cache_version"], sim.LAYER_COMPUTE_CACHE_VERSION)
            self.assertEqual(len(sim.load_layer_compute_cache(str(path))), 1)

    def run_case(self, device, precision, parallelism="TP", degree=1, prefill=True, model="Qwen3_0_6B", comm_mode="serial"):
        seq = 128
        rows = seq if prefill else 64
        if parallelism == "CP":
            rows //= degree
        cache = {}
        standalone_tp = parallelism == "TP" and degree > 1
        if standalone_tp:
            # A complete but stale scalar record must neither bypass the
            # cache-state simulation nor be overwritten with TP results.
            cache_key = (device, model, precision, "prefill" if prefill else "decode",
                         64, seq, prefill and model != "InternVision", parallelism, degree)
            stale_ops = {name: dict(lat_ms=0, dram_access_byte=0, power_w=0)
                         for name in sim.LAYER_COMPUTE_OP_NAMES}
            cache[cache_key] = stale_ops
        calls = []
        real_mm = sim.run_matmul
        def matmul(*args, **kwargs):
            calls.append(tuple(dtype.name for dtype in args[3:7]))
            return real_mm(*args, **kwargs)
        kwargs = dict(model_name=model, model_cfg=sim.test_model_dict[model], precision=precision,
                      pcb=device_dict[device], device=device, M=rows, seq_len_q=seq if prefill else 64,
                      seq_len_kv=seq, is_prefill=prefill, is_causal=prefill and model != "InternVision",
                      parallelism=parallelism, degree=degree,
                      comm_bandwidth_gbps=128 if degree > 1 and comm_mode == "serial" else 0,
                      seq_len=seq, layer_cache_dict=cache, comm_mode=comm_mode)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(sim, "run_matmul", side_effect=matmul), patch.object(sim, "_fill_comm_breakdown_for_layer", wraps=sim._fill_comm_breakdown_for_layer) as comm:
            result = sim.run_layer(**kwargs)
            comm_args = comm.call_args.args
        self.assertGreater(result["lat_comp"], 0)
        self.assertEqual(result["comm_mode"], comm_mode)
        if standalone_tp:
            self.assertEqual(cache, {cache_key: stale_ops})
            self.assertTrue(all(entry["lat_ms"] == 0 for entry in stale_ops.values()))
            op_latencies = result["lat_breakdown_ms"]
            self.assertAlmostEqual(result["lat_comp"], sum(op_latencies.values()))
            communications = [entry for entry in result["execution_sequence"]
                              if entry["kind"] == "communication"]
            self.assertEqual(len(communications), 4)
            self.assertEqual([entry["primitive"] for entry in result["comm_details"].values()].count("ag"), 2)
            self.assertEqual([entry["primitive"] for entry in result["comm_details"].values()].count("rs"), 2)
            self.assertAlmostEqual(sum(entry["lat_ms"] for entry in communications), result["lat_comm"])
            self.assertAlmostEqual(result["execution_sequence"][-1]["end_ms"], result["lat_comp"] + result["lat_comm"])
            self.assertEqual(result["comm_accounting"], "warm_boundary_ag" if comm_mode == "serial" else "ideal_zero_time")
            if comm_mode == "serial":
                self.assertGreater(result["lat_comm"], 0)
        else:
            ops = next(iter(cache.values()))
            op_latencies = {name: entry["lat_ms"] for name, entry in ops.items()}
            self.assertAlmostEqual(result["lat_comp"], sum(value for name, value in op_latencies.items() if name != "qkv_proj_2"))
        self.assertGreater(op_latencies["rope"], 0)
        self.assertEqual(op_latencies["silu"] > 0, model != "InternVision")
        if degree > 1 and comm_mode == "ideal":
            self.assertEqual(result["lat_comm"], 0)
            self.assertEqual(result["power_results_comm"], {})
        if device == "Thor" and precision == "fp8":
            self.assertEqual(calls, [("fp8", "fp8", "fp32", "fp16")] * 5)
            # CP QKV/Attention output FP16; TP RS FP16 and AG FP8.
            self.assertEqual([comm_args[i].name for i in (5, 10, 12, 13, 14, 15)], ["fp16", "fp16", "fp16", "fp8", "fp16", "fp8"])
        if not standalone_tp:
            with contextlib.redirect_stdout(io.StringIO()), patch.object(sim, "run_matmul", side_effect=AssertionError("cache miss")):
                cached = sim.run_layer(**kwargs)
            self.assertAlmostEqual(cached["lat_comp"], result["lat_comp"])
            self.assertAlmostEqual(cached["power_avg_comp"], result["power_avg_comp"])
        return result

    def test_thor_fp8_tp_cp_prefill_decode(self):
        for parallelism, degree, prefill in [("TP", 1, True), ("TP", 2, True), ("TP", 4, True), ("CP", 2, True), ("CP", 4, True), ("TP", 1, False), ("TP", 2, False)]:
            with self.subTest(parallelism=parallelism, degree=degree, prefill=prefill):
                result = self.run_case("Thor", "fp8", parallelism, degree, prefill,
                                       comm_mode="ideal" if parallelism == "CP" else "serial")
                if parallelism == "TP" and degree > 1:
                    ideal = self.run_case("Thor", "fp8", parallelism, degree, prefill, comm_mode="ideal")
                    self.assertEqual(ideal["lat_breakdown_ms"], result["lat_breakdown_ms"])
                    self.assertEqual(ideal["dram_access_breakdown_bytes"], result["dram_access_breakdown_bytes"])
                    for name, actual in result["comm_details"].items():
                        for field in ("tx_bytes", "rx_bytes", "l2_bytes", "dram_bytes"):
                            self.assertEqual(ideal["comm_details"][name][field], actual[field])

    def test_other_precisions(self):
        for device, precision in [("Orin", "fp16"), ("Orin", "int8"), ("Orin", "int4"), ("Thor", "fp4")]:
            with self.subTest(device=device, precision=precision):
                self.run_case(device, precision)

    def test_gelu_model_does_not_halve_up(self):
        self.run_case("Thor", "fp8", model="InternVision")


if __name__ == "__main__":
    unittest.main()
