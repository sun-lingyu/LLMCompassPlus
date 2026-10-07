"""FlashInfer FP16 SiLU/Mul timing with fixed buffers within 80% of L2."""

import argparse
import importlib.metadata
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


def working_set_bytes(m, n):
    # Packed FP16 gate/up input plus FP16 output, with auxiliary headroom.
    return 6 * m * n + 65536


@torch.inference_mode()
def benchmark(m, n, repeats=5):
    from flashinfer.activation import silu_and_mul

    props = torch.cuda.get_device_properties(0)
    l2_bytes = props.L2_cache_size
    estimated_bytes = working_set_bytes(m, n)
    if m <= 0 or n <= 0 or n % 8 or estimated_bytes > 0.8 * l2_bytes:
        raise ValueError(f"{m=} {n=}: working set exceeds 80% of L2")

    x = torch.randn((m, 2 * n), device="cuda", dtype=torch.float16)
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)
    silu_and_mul(x, out=output)
    gate, up = x.float().chunk(2, dim=-1)
    reference = (torch.nn.functional.silu(gate) * up).half()
    torch.testing.assert_close(output, reference, atol=3e-3, rtol=3e-3)
    del gate, up, reference
    for _ in range(3):
        silu_and_mul(x, out=output)
    torch.cuda.synchronize()

    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single):
        silu_and_mul(x, out=output)
    capture_growth = torch.cuda.max_memory_allocated() - allocated_before
    measured_bytes = (x.numel() + output.numel()) * 2 + capture_growth
    if max(estimated_bytes, measured_bytes) > 0.8 * l2_bytes:
        raise ValueError(f"Captured working set exceeds 80% of L2: {measured_bytes}")

    # Queue replays ahead of GPU execution, avoiding CPU launch gaps. The GPU
    # sleep is outside the timed interval and does not touch the working buffers.
    # This runtime cannot capture CUDAGraph.replay inside another graph.
    replays = 256
    for _ in range(20):
        single.replay()
    queued = torch.cuda.Event(enable_timing=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    queued.record()
    start.record()
    end.record()
    torch.cuda.synchronize()
    samples = []
    queue_margins_ms = []
    for _ in range(repeats):
        submitted_at = time.perf_counter()
        queued.record()
        torch.cuda._sleep(50_000_000)
        start.record()
        for _ in range(replays):
            single.replay()
        end.record()
        submission_ms = (time.perf_counter() - submitted_at) * 1000
        end.synchronize()
        margin_ms = queued.elapsed_time(start) - submission_ms
        if margin_ms <= 0:
            raise RuntimeError("CPU did not queue all replays before timing started")
        queue_margins_ms.append(margin_ms)
        samples.append(start.elapsed_time(end) / replays / 1000)
    latency = statistics.median(samples)
    return {
        "M": m,
        "N": n,
        "output_numel": m * n,
        "latency_seconds": latency,
        "samples_seconds": samples,
        "queue_margins_ms": queue_margins_ms,
        "estimated_working_set_bytes": estimated_bytes,
        "measured_working_set_bytes": measured_bytes,
        "l2_bytes": l2_bytes,
        "gpu": props.name,
        "dtype": "fp16",
        "input_dtype": "fp16",
        "output_dtype": "fp16",
        "input_shape": [m, 2 * n],
        "output_shape": [m, n],
        "fused_mul": True,
        "correctness_checked": True,
        "kernel": "flashinfer.activation.silu_and_mul",
        "build_mode": "jit_current_flashinfer_no_aot",
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "graph_nodes": 1,
        "replays_per_sample": replays,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", required=True, help="JSON list of [M, N] pairs")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    os.environ["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    )
    # Same process-local JIT setup as the DRAM benchmark; do not load the old AOT package.
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    from flashinfer.jit import env as jit_env

    with tempfile.TemporaryDirectory(prefix="silu_empty_aot_") as empty_aot:
        jit_env.FLASHINFER_AOT_DIR = Path(empty_aot)
        for m, n in json.loads(args.shapes):
            result = benchmark(m, n, args.repeats)
            print(
                f"M={m} N={n} Average Latency: {result['latency_seconds'] * 1000:.3f} ms",
                flush=True,
            )
            print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
