import argparse
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROPE_ROOT = Path(__file__).resolve().parent / "rope"

try:
    load(
        name="qknorm_rope_ext_impl",
        sources=[
            str(ROPE_ROOT / "csrc" / "bindings.cpp"),
            str(ROPE_ROOT / "csrc" / "fused_qknorm_rope_kernel.cu"),
        ],
        extra_include_paths=[str(ROPE_ROOT / "csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        is_python_module=False,
    )
except Exception as e:
    print(f"Error: Could not build/load fused_qk_norm_rope extension: {e}")
    sys.exit(1)

if not hasattr(torch.ops.qknorm_rope_ext, "fused_qk_norm_rope"):
    print("Error: fused_qk_norm_rope op not registered.")
    sys.exit(1)

fused_qk_norm_rope = torch.ops.qknorm_rope_ext.fused_qk_norm_rope.default

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
CACHE_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
ELEM_SIZE = {"fp16": 2, "bf16": 2, "fp32": 4}


def benchmark_randomized_graph(
    num_tokens,
    head_dim,
    num_heads_q,
    num_heads_kv,
    rotary_dim,
    eps,
    is_neox,
    dtype,
    cache_dtype,
    forced_token_heads_per_warp,
    max_position,
    working_set_mib,
    duration=2000.0,
):
    target_working_set_bytes = working_set_mib * 1024 * 1024

    print("--- Benchmarking fused_qk_norm_rope with a randomized CUDA Graph ---")
    total_heads = num_heads_q + num_heads_kv * 2
    hidden_size = total_heads * head_dim
    print(
        f"num_tokens={num_tokens}, head_dim={head_dim}, "
        f"heads(q/kv)={num_heads_q}/{num_heads_kv}, "
        f"rotary_dim={rotary_dim}, dtype={dtype}, cache_dtype={cache_dtype}, "
        f"is_neox={is_neox}, forced_token_heads_per_warp={forced_token_heads_per_warp}"
    )

    dev = torch.device("cuda:0")
    torch_dtype = DTYPE_MAP[dtype]
    torch_cache_dtype = CACHE_DTYPE_MAP[cache_dtype]
    element_size = ELEM_SIZE[dtype]

    # The allocation contains packed Q/K/V, but this kernel reads and writes
    # only Q and K. Size the cache-defeating working set from touched Q/K bytes.
    qk_heads = num_heads_q + num_heads_kv
    qk_unique_bytes_per_kernel = num_tokens * qk_heads * head_dim * element_size
    qkv_allocated_bytes_per_buffer = num_tokens * hidden_size * element_size
    nodes_needed = math.ceil(target_working_set_bytes / qk_unique_bytes_per_kernel)
    # Large individual tensors need only a few distinct buffers. Avoid forcing
    # a 20-buffer minimum, which can cause unnecessary allocations or OOM.
    nodes_per_graph = max(1, min(nodes_needed, 4000))
    qk_working_set_mib = nodes_per_graph * qk_unique_bytes_per_kernel / (1024**2)
    qkv_pool_mib = nodes_per_graph * qkv_allocated_bytes_per_buffer / (1024**2)
    print(f"Graph Nodes: {nodes_per_graph}")
    print(
        f"Touched Q/K Working Set: {qk_working_set_mib:.2f} MiB "
        f"(target: {working_set_mib:.2f} MiB)"
    )
    print(f"Allocated Packed-QKV Pool: {qkv_pool_mib:.2f} MiB")
    if nodes_needed > 4000:
        print(
            "Warning: the 4000-node cap prevents the touched Q/K working set "
            "from reaching the requested size."
        )

    # QKV pool: this op mutates qkv in-place, so use a large pool of distinct
    # buffers to defeat L2 caching/prefetching (same trick as rmsnorm bench).
    qkv_pool = [
        torch.randn((num_tokens, hidden_size), dtype=torch_dtype, device=dev)
        for _ in range(nodes_per_graph)
    ]

    q_weight = torch.randn(head_dim, dtype=torch_dtype, device=dev)
    k_weight = torch.randn(head_dim, dtype=torch_dtype, device=dev)
    cos_sin_cache = torch.randn(
        max_position, rotary_dim, dtype=torch_cache_dtype, device=dev
    )
    # First-prefill access pattern: positions 0, 1, ..., num_tokens - 1.
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=dev)

    # A seeded shuffle is always a permutation. The former modular stride could
    # revisit only a subset of buffers when stride and pool size were not coprime.
    access_order = list(range(nodes_per_graph))
    random.Random(0).shuffle(access_order)
    assert len(set(access_order)) == nodes_per_graph
    print(
        f"Access Pattern: seeded random permutation "
        f"(first indices: {access_order[:5]}...)"
    )

    def run_graph_nodes():
        for idx in access_order:
            fused_qk_norm_rope(
                qkv_pool[idx],
                num_heads_q,
                num_heads_kv,
                num_heads_kv,
                head_dim,
                eps,
                q_weight,
                k_weight,
                cos_sin_cache,
                is_neox,
                position_ids,
                forced_token_heads_per_warp,
            )

    # Additional warmup (also triggers JIT dispatch resolution).
    with torch.inference_mode():
        run_graph_nodes()
    torch.cuda.synchronize()

    # Capture the fixed, seeded permutation in a CUDA Graph.
    print("Capturing CUDA Graph...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.inference_mode():
            run_graph_nodes()

    # Warmup
    print("Warming up...")
    with torch.inference_mode():
        for _ in range(3):
            g.replay()
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
    print(f"Single Graph Time: {single_graph_time_ms:.3f} ms")
    real_iters = max(1, int(duration / single_graph_time_ms) + 1)

    # Performance Test
    print(f"Replaying Graph ({duration} ms)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(real_iters):
        g.replay()
    end_event.record()
    end_event.synchronize()

    # Calculate Results
    total_time_ms = start_event.elapsed_time(end_event)
    total_kernels = real_iters * nodes_per_graph
    print(f"real_iters: {real_iters}")
    print(f"total_time_ms: {total_time_ms:.1f} ms")
    print(f"per-graph GPU time: {total_time_ms / real_iters:.3f} ms")

    avg_latency_ms = total_time_ms / total_kernels
    # Only Q/K heads are touched by the kernel (read + write).
    total_bytes_per_kernel = 2 * num_tokens * qk_heads * head_dim * element_size
    gib_per_sec = (total_bytes_per_kernel / (avg_latency_ms * 1e-3)) / (1024**3)

    print("\n--- Results ---")
    print(f"Average Latency: {avg_latency_ms:.3f} ms")
    print(f"Effective Bandwidth: {gib_per_sec:.2f} GiB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, help="Number of tokens")
    parser.add_argument(
        "head_dim",
        type=int,
        help="Head dimension",
    )
    parser.add_argument("num_heads_q", type=int, help="Number of query heads")
    parser.add_argument("num_heads_kv", type=int, help="Number of key/value heads")
    parser.add_argument(
        "--rotary_dim",
        type=int,
        default=None,
        help="Rotary dim, default = head_dim (n)",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument(
        "--interleave",
        dest="is_neox",
        action="store_false",
        default=True,
        help="Use interleaved (GPT-J style) RoPE instead of NeoX style",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="bf16",
        help="QKV / weight dtype",
    )
    parser.add_argument(
        "--cache_dtype",
        choices=list(CACHE_DTYPE_MAP.keys()),
        default="fp32",
        help="cos/sin cache dtype",
    )
    parser.add_argument(
        "--forced_token_heads_per_warp",
        type=int,
        choices=[-1, 1, 2, 4, 8],
        default=4,
        help="-1 = auto-select (only on Hopper), else 1/2/4/8",
    )
    parser.add_argument(
        "--max_position", type=int, default=4096, help="Max position for cache"
    )
    parser.add_argument(
        "--working_set_mib",
        type=float,
        default=512.0,
        help="Target unique Q/K bytes touched per graph, in MiB",
    )
    parser.add_argument(
        "--duration", type=float, default=2000.0, help="Minimum duration in ms"
    )
    args = parser.parse_args()

    rotary_dim = args.rotary_dim if args.rotary_dim is not None else args.head_dim
    if args.max_position < args.m:
        parser.error("--max_position must be >= m for first-prefill position_ids")
    if args.working_set_mib <= 0:
        parser.error("--working_set_mib must be positive")

    benchmark_randomized_graph(
        num_tokens=args.m,
        head_dim=args.head_dim,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        rotary_dim=rotary_dim,
        eps=args.eps,
        is_neox=args.is_neox,
        dtype=args.dtype,
        cache_dtype=args.cache_dtype,
        forced_token_heads_per_warp=args.forced_token_heads_per_warp,
        max_position=args.max_position,
        working_set_mib=args.working_set_mib,
        duration=args.duration,
    )
