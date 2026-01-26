import torch
from flash_attn import flash_attn_with_kvcache

# 假设参数配置
batch_size = 16
seqlen_k = 1024  # 当前 KV cache 的长度
nheads_q = 32
nheads_kv = 8
headdim = 128
device = "cuda"
dtype = torch.float16

# 1. 准备 Query
# 解码时 Query 长度通常为 1
q = torch.randn(batch_size, 64, nheads_q, headdim, device=device, dtype=dtype)

# 2. 准备 KV Cache (通常是预分配好的大张量)
# 形状: (batch_size, max_seqlen, nheads, headdim)
# 注意: k_cache 和 v_cache 必须在最后一个维度上是连续的 (contiguous)
k_cache = torch.randn(batch_size, 2048, nheads_kv, headdim, device=device, dtype=dtype)
v_cache = torch.randn(batch_size, 2048, nheads_kv, headdim, device=device, dtype=dtype)

# 3. 准备当前的新 token 的 KV (可选)
# 如果你希望 flash_attn_with_kvcache 帮你把新的 kv 更新进 cache，可以传入 k 和 v
k_new = torch.randn(batch_size, 1, nheads_kv, headdim, device=device, dtype=dtype)
v_new = torch.randn(batch_size, 1, nheads_kv, headdim, device=device, dtype=dtype)

# 4. 准备 cache_seqlens
# 告诉 kernel 每个 batch 实际已经存了多少 token
cache_seqlens = torch.full((batch_size,), seqlen_k, device=device, dtype=torch.int32)

# 5. 调用 flash_attn_with_kvcache 启用 Flash-Decoding
print(f"Before: k_cache at index {seqlen_k} (first batch, first head): {k_cache[0, seqlen_k, 0, :5]}")
print(f"New K (first batch, first head): {k_new[0, 0, 0, :5]}")

output = flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=k_new,              # 传入新 KV 会自动更新到 cache 中 cache_seqlens 的位置
    v=v_new,
    cache_seqlens=cache_seqlens,
    causal=True,          # 解码通常是 causal 的
    num_splits=2          # <--- 关键点: 设置 > 1 启用 Flash-Decoding 并行
)

print("Output shape:", output.shape) # (batch_size, 1, nheads, headdim)
print("Cache sequence length", cache_seqlens) # 应该是原来的 seqlen_k + 1

# 验证 KV Cache 是否更新
# k_new 的 shape 是 (batch_size, 1, nheads, headdim)
# k_cache 的 shape 是 (batch_size, max_seqlen, nheads, headdim)
# 我们检查 k_cache 在 seqlen_k 位置的值是否等于 k_new

# 注意：k_new 是 (batch, 1, ...), k_cache 切片后是 (batch, ...)，需要 squeeze 或者对应维度比较
updated_k = k_cache[:, seqlen_k, :, :]
if torch.allclose(updated_k, k_new.squeeze(1), atol=1e-3):
    print("SUCCESS: KV Cache updated correctly!")
else:
    print("FAILURE: KV Cache NOT updated correctly!")
    print(f"After: k_cache at index {seqlen_k} (first batch, first head): {k_cache[0, seqlen_k, 0, :5]}")
    print(f"Diff: {(updated_k - k_new.squeeze(1)).abs().max()}")

# 检查 cache_seqlens 是否更新 (通常 flash_attn 不会原地更新这个 tensor，需要用户自己维护)
if torch.all(cache_seqlens == seqlen_k):
    print("INFO: cache_seqlens was NOT updated in-place (expected behavior).")
else:
    print("INFO: cache_seqlens WAS updated in-place.")
