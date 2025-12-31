import torch
import argparse
import sys
import time

try:
    import marlin
except ImportError:
    print("Import marlin Error")
    sys.exit(1)

def run_marlin_benchmark(m, n, k, iterations=100, warmup=100):
    print(f"M={m}, N={n}, K={k}")
    
    if k % 128 != 0:
        print("Warning: k % 128 != 0")
    if n % 256 != 0:
        print("Warning: n % 256")

    dev = torch.device('cuda:0')

    # 1. Prepare A (FP16)
    # Shape: [M, K]
    A = torch.randn((m, k), dtype=torch.float16, device=dev)

    # 2. Prepare B (INT4 Packed into INT32)
    B_packed = torch.randint(
        low=-2147483648, high=2147483647, 
        size=(k, n // 16), 
        dtype=torch.int32, 
        device=dev
    )

    # 3. Prepare C (FP16)
    # Shape: [M, N]
    C = torch.zeros((m, n), dtype=torch.float16, device=dev)

    # 4. Prepare Scales (FP16)
    # Marlin requires per-group or per-channel scales。Assume simple per-channel (shape [1, N])
    s = torch.randn((1, n), dtype=torch.float16, device=dev)

    # 5. Prepare Workspace
    workspace = torch.zeros(n // 128 * 16, device=dev, dtype=torch.int)

    # 6. Warmup
    print(f"Warmpup ({warmup} times)...")
    for _ in range(warmup):
        marlin.mul(A, B_packed, C, s, workspace)
    torch.cuda.synchronize()

    # 7. Perf test
    print(f"begin test ({iterations} times)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        marlin.mul(A, B_packed, C, s, workspace)
    end_event.record()
    
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iterations

    # 8. Calculate performance
    # FLOPs = 2 * M * N * K
    total_ops = 2 * m * n * k
    tflops = (total_ops / (avg_time_ms / 1000)) / 1e12

    print(f"average latency: {avg_time_ms:.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marlin Matrix Multiplication Benchmark")
    parser.add_argument("m", type=int, default=16, help="Batch size (M)")
    parser.add_argument("n", type=int, default=4096, help="Output dimension (N)")
    parser.add_argument("k", type=int, default=4096, help="Input dimension (K)")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Torch CUDA Error")
        sys.exit(1)
        
    run_marlin_benchmark(args.m, args.n, args.k, args.iter)