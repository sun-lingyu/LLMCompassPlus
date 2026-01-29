# Run this script with:
#    nsys profile --stats=true --capture-range=cudaProfilerApi --stop-on-exit=true python profile_compare.py

# Latency should be printed in cmd like:
# Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
#     100.0          124,000         10  12,400.0  11,776.0    11,712    17,152      1,693.7  void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, (…

import argparse

import torch
import torch.cuda.profiler as profiler  # 引入手动控制模块
from flash_attn import flash_attn_func

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--nheads_q", type=int, default=32)
parser.add_argument("--nheads_kv", type=int, default=8)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument(
    "--activation_dtype",
    type=str,
    default="fp16",
    choices=["fp16", "bf16", "int32", "fp32", "int8", "int4"],
)
parser.add_argument(
    "--weight_dtype",
    type=str,
    default="fp16",
    choices=["fp16", "bf16", "int32", "fp32", "int8", "int4"],
)
args = parser.parse_args()

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int32": torch.int32,
    "fp32": torch.float32,
    "int8": torch.int8,
    "int4": torch.int8,
}

B = args.batch_size
S = args.seq_len
H_dim = args.head_dim
Q_heads = args.nheads_q
KV_heads = args.nheads_kv

device = torch.device("cuda")
activation_data_type = dtype_map[args.activation_dtype]
weight_data_type = dtype_map[args.weight_dtype]

Q = torch.randn(
    B, S, Q_heads, H_dim, device=device, dtype=activation_data_type
).contiguous()
K = torch.randn(
    B, S, KV_heads, H_dim, device=device, dtype=weight_data_type
).contiguous()
V = torch.randn(
    B, S, KV_heads, H_dim, device=device, dtype=weight_data_type
).contiguous()

print("Preparing Eager Mode...")
# 预热
for _ in range(10):
    flash_attn_func(
        Q,
        K,
        V,
        dropout_p=0.0,
        causal=True,
    )
torch.cuda.synchronize()

print(">>> Profiling Eager Mode (Look for 'Eager' markers)...")
# 1. 开启 NVTX 标记方便定位
torch.cuda.nvtx.range_push("Eager_Mode_Test")
# 2. 开启 nsys 采集
profiler.start()

for _ in range(args.iterations):
    flash_attn_func(
        Q,
        K,
        V,
        dropout_p=0.0,
        causal=True,
    )
# 3. 停止 nsys 采集
profiler.stop()
torch.cuda.nvtx.range_pop()
