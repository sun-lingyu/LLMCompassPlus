from software_model.flashdecoding import FlashAttentionDecode
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse
import torch

if __name__ == "__main__":
    pcb = device_dict["Orin"]
    
    seq_len_q = 64
    seq_len_kv_new = 1
    seq_len_kv_cached = 1023
    head_dim = 128
    num_heads_q = 16
    num_heads_kv = 8
    batch_size = 1
    max_seqlen_kv = 4096
    num_splits = 0
    
    print("FlashDecoding Performance Test")
    
    print(f"Batch size: {batch_size}, Seq len Q: {seq_len_q}, Seq len KV new: {seq_len_kv_new}, Seq len KV cached: {seq_len_kv_cached}, Num heads Q: {num_heads_q}, Num heads KV: {num_heads_kv}, Head dim: {head_dim}, Num splits: {num_splits}")
    
    qkv_data_type = data_type_dict["fp16"]
    output_data_type = data_type_dict["fp16"]
    itermediate_data_type = data_type_dict["fp32"]
    
    q = Tensor((batch_size, num_heads_q, seq_len_q, head_dim), qkv_data_type)
    
    model = FlashAttentionDecode(qkv_data_type=qkv_data_type, output_data_type=output_data_type, intermediate_data_type=itermediate_data_type, pcb_module=pcb, is_causal=True, num_splits=num_splits)
    
    _ = model(
      Tensor((batch_size, num_heads_q, seq_len_q, head_dim), qkv_data_type),
      Tensor((batch_size, num_heads_kv, seq_len_kv_new, head_dim), qkv_data_type),
      Tensor((batch_size, num_heads_kv, seq_len_kv_new, head_dim), qkv_data_type),
      cache_seqlens=seq_len_kv_cached,
      max_seqlen_kv=max_seqlen_kv
    )
    
    real_latency = model.run_on_gpu()
    latency = model.compile_and_simulate(compile_mode="exhaustive")
    roofline_latency = model.roofline_model()
        
    print(f"FlashAttention GPU latency: {real_latency * 1000:.3f} ms")
    
    print(f"FlashAttention latency: {latency*1e3:.3f} ms")
    
    print(f"Roofline model latency: {roofline_latency*1e3:.3f} ms")  