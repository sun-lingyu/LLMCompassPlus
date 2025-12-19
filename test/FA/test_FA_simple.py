from software_model.flashattention import FlashAttention
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    pcb = device_dict["Orin"]
    
    seq_len = 1024
    head_dim = 128
    num_heads_q = 16
    num_heads_kv = 16
    batch_size = 1
    
    print(f"problem: seq_len={seq_len}, head_dim={head_dim}, num_heads_q={num_heads_q}, num_heads_kv={num_heads_kv}, batch_size={batch_size}")
    
    model = FlashAttention(data_type=data_type_dict["fp16"], is_causal=False)
    
    _ = model(
      Tensor((batch_size, num_heads_q, seq_len, head_dim), data_type_dict["fp16"]),
      Tensor((batch_size, num_heads_kv, seq_len, head_dim), data_type_dict["fp16"]),
      Tensor((batch_size, num_heads_kv, seq_len, head_dim), data_type_dict["fp16"]),
    )
    
    
    real_latency = model.run_on_gpu()
    latency = model.compile_and_simulate(pcb_module=pcb, compile_mode="exhaustive")
    roofline_latency = model.roofline_model(pcb_module=pcb)
        
    print(f"FlashAttention GPU latency: {real_latency * 1000:.3f} ms")
    
    print(f"FlashAttention latency: {latency*1e3:.2f} ms")
    
    print(f"Roofline model latency: {roofline_latency*1e3:.2f} ms")  