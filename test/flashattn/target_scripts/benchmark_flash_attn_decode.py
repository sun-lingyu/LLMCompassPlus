import argparse
import random
import sys
import time

import torch
from flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_fa2
from flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_fa3


def benchmark_flash_attn_decode_append(
    flash_attn_with_kvcache,
    b,
    seq_kv_history,
    seq_new,
    hq,
    hkv,
    d,
    num_splits=0,
    duration=1000,
):
    total_cache_capacity = seq_kv_history + seq_new

    print("--- Benchmarking FA3 Decode with Append (KV Cache Update) ---")
    print(f"Batch={b}")
    print(f"History KV Len={seq_kv_history} (Existing)")
    print(f"New Q/K/V Len={seq_new} (Appended)")
    print(f"Heads: Q={hq}, KV={hkv}, Dim={d}")
    print(f"Split-KV: {'Auto' if num_splits == 0 else num_splits}")

    dev = torch.device("cuda:0")
    dtype = torch.float16
    element_size = 2  # fp16

    numel_history = b * seq_kv_history * hkv * d
    numel_new_kv = b * seq_new * hkv * d
    numel_q = b * seq_new * hq * d
    numel_o = b * seq_new * hq * d

    one_call_bytes = (
        numel_q  # Read Q
        + 2 * numel_new_kv  # Read K_new, V_new
        + 2 * numel_history  # Read Cache History (Attn)
        + 2 * numel_new_kv  # Write Cache Update
        + numel_o  # Write Output
    ) * element_size
    print(numel_q * element_size / 1024 / 1024)
    print(
        2 * numel_new_kv  # Read K_new, V_new
        + 2 * numel_history  # Read Cache History (Attn)
        + 2 * numel_new_kv
    )

    target_flush_bytes = 128 * 1024 * 1024  # 128MB to be safe for L2 flushing

    if one_call_bytes > target_flush_bytes:
        nodes_per_graph = 2
    else:
        nodes_per_graph = int(target_flush_bytes / one_call_bytes) + 1

    print(f"Graph Nodes (Pool Size): {nodes_per_graph}")
    print(f"Single Call Est. IO: {one_call_bytes / 1024 / 1024:.2f} MB")

    try:
        # Q: [Batch, Seq_New, Heads_Q, Dim]
        q_pool = [
            torch.randn((b, seq_new, hq, d), dtype=dtype, device=dev)
            for _ in range(nodes_per_graph)
        ]
        k_new_pool = [
            torch.randn((b, seq_new, hkv, d), dtype=dtype, device=dev)
            for _ in range(nodes_per_graph)
        ]
        v_new_pool = [
            torch.randn((b, seq_new, hkv, d), dtype=dtype, device=dev)
            for _ in range(nodes_per_graph)
        ]
        k_cache_pool = [
            torch.randn((b, total_cache_capacity, hkv, d), dtype=dtype, device=dev)
            for _ in range(nodes_per_graph)
        ]
        v_cache_pool = [
            torch.randn((b, total_cache_capacity, hkv, d), dtype=dtype, device=dev)
            for _ in range(nodes_per_graph)
        ]

    except torch.cuda.OutOfMemoryError:
        print("Error: OOM when allocating data pool. Reduce batch size or kv length.")
        sys.exit(1)

    cache_seqlens = torch.full((b,), seq_kv_history, dtype=torch.int32, device=dev)

    access_order = list(range(nodes_per_graph))
    random.shuffle(access_order)

    print("Warming up...")
    with torch.inference_mode():
        for i in range(10):
            idx = access_order[i % nodes_per_graph]
            _ = flash_attn_with_kvcache(
                q=q_pool[idx],
                k_cache=k_cache_pool[idx],
                v_cache=v_cache_pool[idx],
                k=k_new_pool[idx],
                v=v_new_pool[idx],
                cache_seqlens=cache_seqlens,
                causal=True,
                num_splits=num_splits,
            )
    torch.cuda.synchronize()

    print("Capturing Randomized Graph...")
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        with torch.inference_mode():
            for idx in access_order:
                _ = flash_attn_with_kvcache(
                    q=q_pool[idx],
                    k_cache=k_cache_pool[idx],
                    v_cache=v_cache_pool[idx],
                    k=k_new_pool[idx],
                    v=v_new_pool[idx],
                    cache_seqlens=cache_seqlens,
                    causal=True,
                    num_splits=num_splits,
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
    parser = argparse.ArgumentParser(description="FA3 Benchmark (Update KV + Decode)")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "-s",
        "--seqlen_kv",
        type=int,
        default=1024,
        help="Existing History Length in Cache",
    )
    parser.add_argument(
        "-q",
        "--seqlen_q",
        type=int,
        default=64,
        help="New Query/KV Length (Chunk size)",
    )
    parser.add_argument(
        "-hq", "--heads", type=int, default=16, help="Number of Query heads"
    )
    parser.add_argument(
        "-hkv", "--kv_heads", type=int, default=8, help="Number of KV heads"
    )
    parser.add_argument("-d", "--dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--num_splits", type=int, default=0, help="Split-KV factor. 0=Auto"
    )
    parser.add_argument(
        "--duration", type=float, default=1000.0, help="Benchmark duration in ms"
    )

    args = parser.parse_args()

    if args.dim > 256:
        print("Error: HeadDim > 256 is not supported by FlashAttention.")
        sys.exit(1)

    avg_latency_ms_fa2 = benchmark_flash_attn_decode_append(
        flash_attn_with_kvcache_fa2,
        args.batch,
        args.seqlen_kv,
        args.seqlen_q,
        args.heads,
        args.kv_heads,
        args.dim,
        num_splits=args.num_splits,
        duration=args.duration,
    )

    avg_latency_ms_fa3 = benchmark_flash_attn_decode_append(
        flash_attn_with_kvcache_fa3,
        args.batch,
        args.seqlen_kv,
        args.seqlen_q,
        args.heads,
        args.kv_heads,
        args.dim,
        num_splits=args.num_splits,
        duration=args.duration,
    )

    print(f"FA2: {avg_latency_ms_fa2}ms")
    print(f"FA3: {avg_latency_ms_fa3}ms")
    print(f"Average Latency: {min(avg_latency_ms_fa2, avg_latency_ms_fa3):.4f} ms")
