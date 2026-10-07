"""FP16 residual + RMSNorm timing with a fixed, L2-sized working set."""

import argparse
import json
import statistics
import time

import torch
from flash_attn.ops.rms_norm import DropoutAddRMSNorm


def working_set_bytes(m, n):
    # Two inputs, two outputs, weight, row statistics and auxiliary headroom.
    return 8 * m * n + 2 * n + 16 * m + 65536


@torch.inference_mode()
def benchmark(m, n, repeats=5):
    props = torch.cuda.get_device_properties(0)
    l2_bytes = props.L2_cache_size
    estimated_bytes = working_set_bytes(m, n)
    if m <= 0 or n <= 0 or estimated_bytes > 0.8 * l2_bytes:
        raise ValueError(f"{m=} {n=}: working set exceeds 80% of L2")

    x = torch.randn((1, m, n), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    model = DropoutAddRMSNorm(
        hidden_size=n, eps=1e-5, device="cuda", dtype=torch.float16, prenorm=True
    ).eval()
    # Initialize the kernel outside capture, on a side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            warmup_output = model(x, residual=residual)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    del warmup_output

    # Capture once: every replay uses the same inputs, outputs and scratch space.
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single):
        outputs = model(x, residual=residual)
    capture_growth = torch.cuda.max_memory_allocated() - allocated_before
    tensor_bytes = x.numel() * x.element_size() * 2 + model.weight.numel() * 2
    measured_bytes = tensor_bytes + capture_growth
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
    # Keep graph outputs alive through timing.
    assert len(outputs) == 2
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
        "kernel": "DropoutAddRMSNorm(prenorm=True)",
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
    for m, n in json.loads(args.shapes):
        result = benchmark(m, n, args.repeats)
        print(
            f"M={m} N={n} Average Latency: {result['latency_seconds'] * 1000:.3f} ms",
            flush=True,
        )
        print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
