import torch
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention

if __name__ == "__main__":
    batch_size = 1
    seq_len = 64
    num_heads_q = 16
    num_heads_kv = 16
    head_dim = 128

    q = torch.randn(
        batch_size, seq_len, num_heads_q, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads_kv, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads_kv, head_dim, device="cuda", dtype=torch.float16
    )

    output = flash_attn_func(q, k, v, causal=False, dropout_p=0.0, softmax_scale=None)

    # scaled_dot_product_attention expects (batch, heads, seq_len, head_dim)
    q_pt = q.transpose(1, 2)
    k_pt = k.transpose(1, 2)
    v_pt = v.transpose(1, 2)

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output2 = scaled_dot_product_attention(
            q_pt, k_pt, v_pt, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    # Transpose back to (batch, seq_len, heads, head_dim)
    output2 = output2.transpose(1, 2)

    print(
        "Difference between flash_attn_func and scaled_dot_product_attention:",
        torch.max(torch.abs(output - output2)).item(),
    )
