import argparse

from hardware_model.device import device_dict
from software_model.flashattn import FlashAttn
from software_model.utils import Tensor, data_type_dict
from test.flashattn.utils import get_output_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument("seq_len_q", type=int)
    parser.add_argument("seq_len_kv", type=int)
    parser.add_argument("head_dim", type=int)
    parser.add_argument("num_heads_q", type=int)
    parser.add_argument("num_heads_kv", type=int)
    parser.add_argument(
        "precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()
    seq_len_q = args.seq_len_q
    seq_len_kv = args.seq_len_kv
    head_dim = args.head_dim
    num_heads_q = args.num_heads_q
    num_heads_kv = args.num_heads_kv
    print(f"problem: seq_len_q {seq_len_q} seq_len_kv {seq_len_kv}")

    pcb = device_dict[args.device]

    if args.precision in ("fp16", "int8", "int4"):
        qkv_dtype = data_type_dict["fp16"]
        intermediate_dtype = data_type_dict["fp32"]
    elif args.precision in ("fp8", "fp4"):
        qkv_dtype = data_type_dict["fp8"]
        intermediate_dtype = data_type_dict["fp32"]
    output_dtype = get_output_dtype(args.precision, True)

    model = FlashAttn(
        qkv_dtype=qkv_dtype,
        intermediate_dtype=intermediate_dtype,
        output_dtype=output_dtype,
        is_causal=True,
        num_splits=0,
        device=args.device,
    )
    _ = model(
        Tensor([seq_len_q, num_heads_q, head_dim], dtype=qkv_dtype),
        Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
        Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
    )

    latency = (
        model.compile_and_simulate(pcb, is_prefill=True)
        + pcb.compute_module.launch_latency.flashattn
    )

    roofline_latency = model.roofline_model(pcb)

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")
