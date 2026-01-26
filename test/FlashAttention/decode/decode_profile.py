import torch
import torch.cuda.profiler as profiler  
from flash_attn import flash_attn_with_kvcache
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nheads_q', type=int, default=8)
parser.add_argument('--nheads_kv', type=int, default=8)
parser.add_argument('--headdim', type=int, default=128)
parser.add_argument('--seqlen_q', type=int, default=64)
parser.add_argument('--seqlen_kv', type=int, default=4095)
parser.add_argument('--seqlen_kv_new', type=int, default=1)
parser.add_argument('--max_seqlen_kv', type=int, default=4096)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--num_splits', type=int, default=0)
parser.add_argument('--activation_dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'int32', 'fp32', 'int8', 'int4'])
parser.add_argument('--weight_dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'int32', 'fp32', 'int8', 'int4'])
args = parser.parse_args()

dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'int32': torch.int32, 'fp32': torch.float32, 'int8': torch.int8, 'int4': torch.int8} # int4 mapped to int8 as PyTorch does not have int4 dtype
batch_size = args.batch_size
seqlen_q = args.seqlen_q
seqlen_kv = args.seqlen_kv
seqlen_kv_new = args.seqlen_kv_new
nheads_q = args.nheads_q
nheads_kv = args.nheads_kv
headdim = args.headdim
num_splits = args.num_splits
max_seqlen_kv = args.max_seqlen_kv
iterations = args.iterations
activation_dtype = dtype_map[args.activation_dtype]
weight_dtype = dtype_map[args.weight_dtype]

device = "cuda"
# 1. 准备 Query
# 解码时 Query 长度通常为 1
q = torch.randn(batch_size, seqlen_q, nheads_q, headdim, device=device, dtype=activation_dtype)

# 2. 准备 KV Cache (通常是预分配好的大张量)
# 形状: (batch_size, max_seqlen, nheads, headdim)
# 注意: k_cache 和 v_cache 必须在最后一个维度上是连续的 (contiguous)
k_cache = torch.randn(batch_size, max_seqlen_kv, nheads_kv, headdim, device=device, dtype=weight_dtype)
v_cache = torch.randn(batch_size, max_seqlen_kv, nheads_kv, headdim, device=device, dtype=weight_dtype)

# 3. 准备当前的新 token 的 KV (可选)
# 如果你希望 flash_attn_with_kvcache 帮你把新的 kv 更新进 cache，可以传入 k 和 v
k_new = torch.randn(batch_size, seqlen_kv_new, nheads_kv, headdim, device=device, dtype=weight_dtype)
v_new = torch.randn(batch_size, seqlen_kv_new, nheads_kv, headdim, device=device, dtype=weight_dtype)

# 4. 准备 cache_seqlens
# 告诉 kernel 每个 batch 实际已经存了多少 token
cache_seqlens = torch.full((batch_size,), seqlen_kv, device=device, dtype=torch.int32)

print("Preparing Flash-Decoding with KV Cache...")

for _ in range(10): # 预热
    _ = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,              # 传入新 KV 会自动更新到 cache 中 cache_seqlens 的位置
        v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,          # 解码通常是 causal 的
        num_splits=num_splits          # <--- 关键点: 设置 > 1 启用 Flash-Decoding 并行
    )
torch.cuda.synchronize()

print(">>> Profiling Flash-Decoding with KV Cache...")

torch.cuda.nvtx.range_push("Flash_Decoding_KVCache_Test")

profiler.start()

for _ in range(iterations):
    output = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,              # 传入新 KV 会自动更新到 cache 中 cache_seqlens 的位置
        v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,          # 解码通常是 causal 的
        num_splits=num_splits          # <--- 关键点: 设置 > 1 启用 Flash-Decoding 并行
    )
    
torch.cuda.synchronize()
    
profiler.stop()

torch.cuda.nvtx.range_pop()

print("Output shape:", output.shape) # (batch_size, 1, nheads, headdim)
