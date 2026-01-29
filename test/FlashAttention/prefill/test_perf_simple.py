from hardware_model.device import device_dict
from software_model.flashattention import FlashAttentionPrefill
from software_model.utils import Tensor, data_type_dict

if __name__ == "__main__":
    pcb = device_dict["Orin"]

    seq_len = 3072
    head_dim = 128
    num_heads_q = 32
    num_heads_kv = 8
    batch_size = 1

    print(
        f"problem: seq_len={seq_len}, head_dim={head_dim}, num_heads_q={num_heads_q}, num_heads_kv={num_heads_kv}, batch_size={batch_size}"
    )
    qkv_data_type = data_type_dict["fp16"]
    output_data_type = data_type_dict["fp16"]
    intermediate_data_type = data_type_dict["fp32"]

    model = FlashAttentionPrefill(
        qkv_data_type=qkv_data_type,
        output_data_type=output_data_type,
        intermediate_data_type=intermediate_data_type,
        pcb_module=pcb,
        is_causal=True,
    )

    _ = model(
        Tensor((batch_size, num_heads_q, seq_len, head_dim), qkv_data_type),
        Tensor((batch_size, num_heads_kv, seq_len, head_dim), qkv_data_type),
        Tensor((batch_size, num_heads_kv, seq_len, head_dim), qkv_data_type),
    )

    real_latency = model.run_on_gpu()
    latency = model.compile_and_simulate(compile_mode="exhaustive")
    roofline_latency = model.roofline_model()

    print(f"FlashAttention GPU latency: {real_latency * 1e3:.3f} ms")

    print(f"FlashAttention latency: {latency * 1e3:.3f} ms")

    print(f"Roofline model latency: {roofline_latency * 1e3:.3f} ms")
