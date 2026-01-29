import argparse
import sys
import time

import torch

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
except ImportError:
    print("Error: Could not import rms_norm_fn. Ensure flash_attn is installed.")
    sys.exit(1)


def benchmark_with_cuda_graphs(m, n, duration=1000):
    print("--- Benchmarking with CUDA Graphs ---")
    print(f"Shape: M={m}, N={n}, Batch=1, FP16")

    dev = torch.device("cuda:0")
    dtype = torch.float16

    static_x = torch.randn((1, m, n), dtype=dtype, device=dev)
    static_res = torch.randn((1, m, n), dtype=dtype, device=dev)
    static_weight = torch.ones((n,), dtype=dtype, device=dev)
    static_bias = None

    print("Warming up Triton kernel...")
    for _ in range(10):
        rms_norm_fn(
            static_x,
            static_weight,
            static_bias,
            residual=static_res,
            eps=1e-5,
            prenorm=True,
        )
    torch.cuda.synchronize()

    print("Capturing CUDA Graph...")
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        out, new_res = rms_norm_fn(
            static_x,
            static_weight,
            static_bias,
            residual=static_res,
            eps=1e-5,
            prenorm=True,
        )

    print(f"Replaying Graph {duration} ms...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_cpu = time.time()
    start_event.record()

    iterations = 0
    while True:
        g.replay()
        iterations += 1

        if iterations % 1000 == 0:
            if (time.time() - start_cpu) * 1000 >= duration:
                break

    end_event.record()
    end_event.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / iterations

    print("\n--- Results (CUDA Graphs) ---")
    print(f"Average Latency: {avg_latency_ms:.4f} ms")

    if avg_latency_ms < 5e-3:
        print(
            "\n[Note] Latency is extremely low (<5us). This is typical for small kernels executed via Graphs."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "m",
        type=int,
        default=16,
        help="Sequence Length (M) - Try small value like 1 or 16",
    )
    parser.add_argument("n", type=int, default=4096, help="Hidden Size (N)")
    parser.add_argument("--duration", type=float, default=1000, help="Duration in ms")
    args = parser.parse_args()

    benchmark_with_cuda_graphs(args.m, args.n, args.duration)
