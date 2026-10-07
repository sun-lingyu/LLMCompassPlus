"""BF16 fused QKNorm/RoPE timing with fixed buffers fitting within 80% of L2."""

import argparse
import json
import os
import statistics
import sys
import time

import torch


def working_set_bytes(m, head_dim, heads_q, heads_kv):
    # Conservatively include packed V, despite the kernel touching only Q/K.
    # cos/sin rows correspond exactly to positions 0..M-1; FP32 cache, BF16 weights.
    return (
        m * (2 * (heads_q + 2 * heads_kv) * head_dim + 4 * head_dim + 8)
        + 4 * head_dim
        + 65536
    )


@torch.inference_mode()
def benchmark(m, head_dim, heads_q, heads_kv, kernel, repeats=5):
    props = torch.cuda.get_device_properties(0)
    l2_bytes = props.L2_cache_size
    estimated_bytes = working_set_bytes(m, head_dim, heads_q, heads_kv)
    if min(m, head_dim, heads_q, heads_kv) <= 0 or estimated_bytes > 0.8 * l2_bytes:
        raise ValueError("Invalid shape or working set exceeds 80% of L2")
    qkv = torch.randn(
        (m, (heads_q + 2 * heads_kv) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn_like(q_weight)
    cos_sin = torch.randn((m, head_dim), device="cuda", dtype=torch.float32)
    positions = torch.arange(m, device="cuda", dtype=torch.int64)

    def run():
        # Preserve existing benchmark defaults: full rotary dimension, NeoX,
        # eps=1e-6 and four token-heads per warp. QKV is updated in place.
        kernel(
            qkv,
            heads_q,
            heads_kv,
            heads_kv,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            cos_sin,
            True,
            positions,
            4,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single):
        run()
    capture_growth = torch.cuda.max_memory_allocated() - allocated_before
    measured_bytes = (
        sum(
            t.numel() * t.element_size()
            for t in (qkv, q_weight, k_weight, cos_sin, positions)
        )
        + capture_growth
    )
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
        "N": (heads_q + heads_kv) * head_dim,
        "head_dim": head_dim,
        "num_heads_q": heads_q,
        "num_heads_kv": heads_kv,
        "output_numel": m * (heads_q + heads_kv) * head_dim,
        "latency_seconds": latency,
        "samples_seconds": samples,
        "queue_margins_ms": queue_margins_ms,
        "estimated_working_set_bytes": estimated_bytes,
        "measured_working_set_bytes": measured_bytes,
        "l2_bytes": l2_bytes,
        "gpu": props.name,
        "dtype": "bf16",
        "cache_dtype": "fp32",
        "eps": 1e-6,
        "is_neox": True,
        "forced_token_heads_per_warp": 4,
        "kernel": "fused_qk_norm_rope",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "graph_nodes": 1,
        "replays_per_sample": replays,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", required=True, help="JSON list of [M, head_dim, heads_q, heads_kv]"
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    # Reuse the existing deployed benchmark's extension loader and CUDA kernel.
    os.environ["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    )
    from benchmark_fused_qknorm_rope import fused_qk_norm_rope

    for m, head_dim, heads_q, heads_kv in json.loads(args.shapes):
        result = benchmark(
            m, head_dim, heads_q, heads_kv, fused_qk_norm_rope, args.repeats
        )
        print(
            f"M={m} head_dim={head_dim} heads={heads_q}/{heads_kv} Average Latency: {result['latency_seconds'] * 1000:.3f} ms",
            flush=True,
        )
        print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
