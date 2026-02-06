import argparse
import random
import sys
import time

import torch

try:
    import marlin
except ImportError:
    print("Import marlin Error")
    sys.exit(1)


def run_marlin_benchmark(m, n, k, iterations=100, warmup=100, duration=0):
    total_bytes = 64 * 1024 * 1024  # 64MB, need to be larger than L2 cache size

    print(f"M={m}, N={n}, K={k}")

    if k % 128 != 0:
        print("Warning: k % 128 != 0")
    if n % 256 != 0:
        print("Warning: n % 256")

    dev = torch.device("cuda:0")

    num_copies_A = int(total_bytes / (m * k * 2))  # fp16
    num_copies_B = int(total_bytes / (k * n // 2))  # int4
    num_copies_C = int(total_bytes / (m * n * 2))  # fp16
    access_indices_A = list(range(num_copies_A))
    random.shuffle(access_indices_A)
    access_indices_B = list(range(num_copies_B))
    random.shuffle(access_indices_B)
    access_indices_C = list(range(num_copies_C))
    random.shuffle(access_indices_C)

    A_list, B_list, C_list = [], [], []
    # 1. Prepare A
    for _ in range(num_copies_A):
        A_list.append(torch.randn((m, k), dtype=torch.float16, device=dev))

    # 2. Prepare B
    for _ in range(num_copies_B):
        B_list.append(
            torch.randint(
                low=-2147483648,
                high=2147483647,
                size=(k, n // 16),
                dtype=torch.int32,
                device=dev,
            )
        )
    # 3. Prepare C
    for _ in range(num_copies_C):
        C_list.append(torch.zeros((m, n), dtype=torch.float16, device=dev))

    # 4. Prepare Scales (FP16)
    # Marlin requires per-group or per-channel scales。Assume simple per-channel (shape [1, N])
    s = torch.randn((1, n), dtype=torch.float16, device=dev)

    # 5. Prepare Workspace
    workspace = torch.zeros(n // 128 * 16, device=dev, dtype=torch.int)

    # 6. Warmup
    print(f"Warmpup ({warmup} times)...")
    for i in range(warmup):
        rand_idx_A = access_indices_A[i % num_copies_A]
        rand_idx_B = access_indices_B[i % num_copies_B]
        rand_idx_C = access_indices_C[i % num_copies_C]
        marlin.mul(
            A_list[rand_idx_A], B_list[rand_idx_B], C_list[rand_idx_C], s, workspace
        )
    torch.cuda.synchronize()

    # 7. Perf test
    if duration > 0:
        print(f"begin test (min ({iterations} times, {duration} ms))...")
    else:
        print(f"begin test ({iterations} times)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    i = 0
    start_cpu = time.time()

    start_event.record()
    while True:
        rand_idx_A = access_indices_A[i % num_copies_A]
        rand_idx_B = access_indices_B[i % num_copies_B]
        rand_idx_C = access_indices_C[i % num_copies_C]
        marlin.mul(
            A_list[rand_idx_A], B_list[rand_idx_B], C_list[rand_idx_C], s, workspace
        )
        i += 1
        if i >= iterations:
            current_cpu = time.time()
            if duration <= 0 or (current_cpu - start_cpu) * 1000 >= duration:
                break
    end_event.record()
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / i

    # 8. Calculate performance
    # FLOPs = 2 * M * N * K
    total_ops = 2 * m * n * k
    tflops = (total_ops / (avg_time_ms / 1000)) / 1e12

    print(f"average latency: {avg_time_ms:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Marlin Matrix Multiplication Benchmark"
    )
    parser.add_argument("m", type=int, default=16, help="Batch size (M)")
    parser.add_argument("n", type=int, default=4096, help="Output dimension (N)")
    parser.add_argument("k", type=int, default=4096, help="Input dimension (K)")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Minimum duration in ms (default: 0)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Torch CUDA Error")
        sys.exit(1)

    run_marlin_benchmark(args.m, args.n, args.k, args.iter, duration=args.duration)
