"""FlashInfer SiLU/Mul, FP16 input/output, randomized large-working-set timing."""

import argparse
import importlib.metadata
import json
import math
import os
import random
import statistics
import sys
import tempfile

import torch


@torch.inference_mode()
def benchmark(m, n, duration_ms=500, repeats=3, working_set_mib=512):
    from flashinfer.activation import silu_and_mul

    if m <= 0 or n <= 0 or n % 8:
        raise ValueError(
            "M must be positive; output width N must be a positive multiple of 8"
        )
    # Each buffer has packed [gate, up] input and a distinct FP16 output.
    bytes_per_call = m * n * 6
    nodes = min(4000, max(1, math.ceil(working_set_mib * 1024**2 / bytes_per_call)))
    actual_bytes = nodes * bytes_per_call
    props = torch.cuda.get_device_properties(0)
    if actual_bytes < 4 * props.L2_cache_size:
        raise ValueError(
            "Working set too small for DRAM timing; increase the shape or working set"
        )
    inputs = [
        torch.randn((m, 2 * n), device="cuda", dtype=torch.float16)
        for _ in range(nodes)
    ]
    outputs = [
        torch.empty((m, n), device="cuda", dtype=torch.float16) for _ in range(nodes)
    ]
    order = list(range(nodes))
    random.Random(0).shuffle(order)

    # Compile and check one call outside the timed region.
    silu_and_mul(inputs[0], out=outputs[0])
    gate, up = inputs[0].float().chunk(2, dim=-1)
    reference = (torch.nn.functional.silu(gate) * up).half()
    torch.testing.assert_close(outputs[0], reference, atol=3e-3, rtol=3e-3)
    del gate, up, reference

    def run():
        for idx in order:
            silu_and_mul(inputs[idx], out=outputs[idx])

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def elapsed(replays):
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    replays = max(1, math.ceil(duration_ms / elapsed(1)))
    samples = [elapsed(replays) / (replays * nodes) for _ in range(repeats)]
    latency_ms = statistics.median(samples)
    return {
        "M": m,
        "N": n,
        "input_shape": [m, 2 * n],
        "output_shape": [m, n],
        "input_dtype": "fp16",
        "output_dtype": "fp16",
        "fused_mul": True,
        "kernel": "flashinfer.activation.silu_and_mul",
        "build_mode": "jit_current_flashinfer_no_aot",
        "latency_ms": latency_ms,
        "samples_ms": samples,
        "working_set_bytes": actual_bytes,
        "target_working_set_mib": working_set_mib,
        "graph_nodes": nodes,
        "replays_per_sample": replays,
        "l2_bytes": props.L2_cache_size,
        "gpu": props.name,
        "effective_bandwidth_GBps": bytes_per_call / (latency_ms * 1e6),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "correctness_checked": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", required=True, help="JSON list of [M, output N]")
    parser.add_argument("--duration-ms", type=float, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--working-set-mib", type=float, default=512)
    args = parser.parse_args()
    if args.duration_ms <= 0 or args.repeats < 1 or args.working_set_mib <= 0:
        parser.error("duration, repeats and working set must be positive")
    os.environ["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    )
    # Target machines have FlashInfer 0.6.4 with an older 0.5.3 AOT package.
    # Ignore that package in this process and JIT the installed Python version.
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    from flashinfer.jit import env as jit_env

    with tempfile.TemporaryDirectory(prefix="silu_empty_aot_") as empty_aot:
        jit_env.FLASHINFER_AOT_DIR = type(jit_env.FLASHINFER_AOT_DIR)(empty_aot)
        for m, n in json.loads(args.shapes):
            result = benchmark(
                m, n, args.duration_ms, args.repeats, args.working_set_mib
            )
            print(
                f"M={m} N={n} Average Latency: {result['latency_ms']:.3f} ms",
                flush=True,
            )
            print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
