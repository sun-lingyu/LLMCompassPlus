import argparse
import sys
import time

import torch

try:
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except ImportError:
    print("Error: Could not import RMSNorm.")
    sys.exit(1)


def benchmark_randomized_graph(m, n, duration=1000):
    target_flush_bytes = 512 * 1024 * 1024  # 512 MB >> L2 cache size

    print("--- Benchmarking with Randomized CUDA Graphs ---")
    print(f"Shape: M={m}, N={n}, Batch=1, FP16")

    dev = torch.device("cuda:0")
    dtype = torch.float16
    element_size = 2

    # (Read X + Read Res)
    pair_bytes = m * n * element_size * 2
    nodes_per_graph = int(target_flush_bytes / pair_bytes) + 1
    nodes_per_graph = max(100, min(nodes_per_graph, 4000))
    actual_mb = (nodes_per_graph * pair_bytes) / (1024**2)
    print(f"Graph Nodes: {nodes_per_graph}")
    print(f"Total Working Set: {actual_mb:.2f} MB (Target > L2 Cache)")

    x_pool = [
        torch.randn((1, m, n), dtype=dtype, device=dev) for _ in range(nodes_per_graph)
    ]
    res_pool = [
        torch.randn((1, m, n), dtype=dtype, device=dev) for _ in range(nodes_per_graph)
    ]
    model = DropoutAddRMSNorm(
        hidden_size=n, eps=1e-5, device=dev, dtype=dtype, prenorm=True
    )
    model.eval()
    stride = nodes_per_graph // 2 + 1
    access_order = [(i * stride) % nodes_per_graph for i in range(nodes_per_graph)]
    print(f"Access Pattern: Stride (e.g., {access_order[:5]}...) to defeat prefetcher.")

    def run_graph_nodes():
        outs = []
        for idx in access_order:
            # Must keep return value to avoid Python GC immediately recycling it.
            res = model(x_pool[idx], residual=res_pool[idx])
            outs.append(res)
        return outs

    # Additional warmup for Triton
    with torch.inference_mode():
        warmup_outs = run_graph_nodes()
    del warmup_outs
    torch.cuda.synchronize()

    # Capture Randomized Graph
    print("Capturing Randomized Graph...")
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        with torch.inference_mode():
            captured_outs = run_graph_nodes()

    # Warmup
    print("Warming up...")
    with torch.inference_mode():
        for _ in range(3):
            warmup_outs = g.replay()
    del warmup_outs
    torch.cuda.synchronize()

    # Estimate iterations for target duration
    print("Estimating iterations for target duration...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    g.replay()
    end_event.record()
    end_event.synchronize()
    single_graph_time_ms = start_event.elapsed_time(end_event)
    print(f"Single Graph Time: {single_graph_time_ms:.1f} ms")
    real_iters = max(1, int(duration / single_graph_time_ms) + 1)

    # Performance Test
    print(f"Replaying Graph ({duration} ms)...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_cpu = time.time()
    start_event.record()
    for i in range(real_iters):
        g.replay()
        torch.cuda.empty_cache()
    end_event.record()
    end_event.synchronize()

    # Calculate Results
    total_time_ms = start_event.elapsed_time(end_event)
    total_kernels = real_iters * nodes_per_graph
    print(f"real_iters: {real_iters}")
    print(f"total_time_ms: {total_time_ms:.1f} ms")
    print(f"per-graph GPU time (if sync correct): {total_time_ms / real_iters:.3f} ms")

    avg_latency_ms = total_time_ms / total_kernels
    # Bandwidth Calculation (4 IOs)
    total_bytes_per_kernel = 4 * m * n * 2
    gb_per_sec = (total_bytes_per_kernel / (avg_latency_ms * 1e-3)) / (1024**3)

    print("\n--- Results ---")
    print(f"Average Latency: {avg_latency_ms:.3f} ms")
    print(f"Effective Bandwidth: {gb_per_sec:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, default=1, help="Sequence Length (M)")
    parser.add_argument("n", type=int, default=4096, help="Hidden Size (N)")
    parser.add_argument(
        "--duration", type=float, default=2000.0, help="Minimum duration in ms"
    )
    args = parser.parse_args()

    benchmark_randomized_graph(args.m, args.n, args.duration)
