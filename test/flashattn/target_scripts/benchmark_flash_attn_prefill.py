import argparse
import random
import sys
import time

import torch
from flash_attn import flash_attn_func_fa2
from flash_attn_interface import flash_attn_func_fa3


def benchmark_flash_attn_graph(
    flash_attn_func,
    b,
    s,
    hq,
    hkv,
    d,
    num_splits=1,
    causal=False,
    pack_gqa=False,
    duration=1000,
):
    print("--- Benchmarking FlashAttention with Randomized CUDA Graphs ---")
    print(
        f"Shape: Batch={b}, SeqLen={s}, Q_Heads={hq}, KV_Heads={hkv}, HeadDim={d}, Causal={causal}"
    )

    dev = torch.device("cuda:0")
    dtype = torch.float16
    element_size = 2  # fp16

    numel_q = b * s * hq * d
    numel_kv = b * s * hkv * d
    numel_o = b * s * hq * d

    one_call_bytes = (numel_q + 2 * numel_kv + numel_o) * element_size

    target_flush_bytes = 64 * 1024 * 1024
    nodes_per_graph = int(target_flush_bytes / one_call_bytes) + 1

    print(f"nodes_per_graph: {nodes_per_graph}")

    actual_mb = (nodes_per_graph * one_call_bytes) / (1024**2)
    print(f"Graph Nodes: {nodes_per_graph}")
    print(f"Total Working Set: {actual_mb:.2f} MB (Target > L2 Cache)")

    q_pool = [
        torch.randn((b, s, hq, d), dtype=dtype, device=dev)
        for _ in range(nodes_per_graph)
    ]
    k_pool = [
        torch.randn((b, s, hkv, d), dtype=dtype, device=dev)
        for _ in range(nodes_per_graph)
    ]
    v_pool = [
        torch.randn((b, s, hkv, d), dtype=dtype, device=dev)
        for _ in range(nodes_per_graph)
    ]

    access_order = list(range(nodes_per_graph))
    random.shuffle(access_order)

    # 4. Warmup
    print("Warming up...")
    with torch.inference_mode():
        for i in range(10):
            rand_idx = access_order[i % nodes_per_graph]
            _ = flash_attn_func(
                q_pool[rand_idx],
                k_pool[rand_idx],
                v_pool[rand_idx],
                num_splits=num_splits,
                causal=causal,
                pack_gqa=pack_gqa,
            )
    torch.cuda.synchronize()

    print("Capturing Randomized Graph...")
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        with torch.inference_mode():
            for idx in access_order:
                _ = flash_attn_func(
                    q_pool[idx],
                    k_pool[idx],
                    v_pool[idx],
                    num_splits=num_splits,
                    causal=causal,
                    pack_gqa=pack_gqa,
                )

    print(f"Replaying Graph ({duration} ms)...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_cpu = time.time()
    start_event.record()

    real_iters = 0
    while True:
        g.replay()
        real_iters += 1
        if real_iters % 10 == 0:
            if (time.time() - start_cpu) * 1000 >= duration:
                break

    end_event.record()
    end_event.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    total_kernels = real_iters * nodes_per_graph
    avg_latency_ms = total_time_ms / total_kernels

    return avg_latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention CUDA Graph Benchmark")
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-s", "--seqlen", type=int, default=1024)
    parser.add_argument("-hq", "--heads", type=int, default=16)
    parser.add_argument("-hkv", "--kv_heads", type=int, default=16)
    parser.add_argument("-d", "--dim", type=int, default=128)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--pack_gqa", action="store_true")
    parser.add_argument(
        "--duration", type=float, default=1000.0, help="Benchmark duration in ms"
    )

    args = parser.parse_args()

    if args.dim > 256:
        print("Error: HeadDim > 256 is not supported by FlashAttention.")
        sys.exit(1)

    avg_latency_ms_fa2 = benchmark_flash_attn_graph(
        flash_attn_func_fa2,
        args.batch,
        args.seqlen,
        args.heads,
        args.kv_heads,
        args.dim,
        num_splits=args.num_splits,
        causal=args.causal,
        pack_gqa=args.pack_gqa,
        duration=args.duration,
    )

    avg_latency_ms_fa3 = benchmark_flash_attn_graph(
        flash_attn_func_fa3,
        args.batch,
        args.seqlen,
        args.heads,
        args.kv_heads,
        args.dim,
        num_splits=args.num_splits,
        causal=args.causal,
        pack_gqa=args.pack_gqa,
        duration=args.duration,
    )
    print(f"FA2: {avg_latency_ms_fa2}ms")
    print(f"FA3: {avg_latency_ms_fa3}ms")
    print(f"Average Latency: {min(avg_latency_ms_fa2, avg_latency_ms_fa3):.4f} ms")
