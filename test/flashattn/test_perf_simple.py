import argparse

from hardware_model.device import device_dict
from software_model.flashattn import FlashAttn
from software_model.flashattn_combine import FlashAttentionCombine
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
    parser.add_argument("num_splits", type=int)
    parser.add_argument(
        "precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    parser.add_argument("--is_prefill", action="store_true")
    parser.add_argument("--is_causal", action="store_true")
    args = parser.parse_args()
    seq_len_q = args.seq_len_q
    seq_len_kv = args.seq_len_kv
    head_dim = args.head_dim
    num_heads_q = args.num_heads_q
    num_heads_kv = args.num_heads_kv
    print(f"problem: seq_len_q {seq_len_q} seq_len_kv {seq_len_kv}")
    print(f"head_dim: {head_dim}, num_splits: {args.num_splits}")
    print(f"q_heads: {args.num_heads_q}, kv_heads: {args.num_heads_kv}")
    print(f"is_prefill: {args.is_prefill}, is_causal: {args.is_causal}")
    if args.is_prefill:
        assert args.num_splits == 1

    pcb = device_dict[args.device]

    if args.precision in ("fp16", "int8", "int4"):
        qkv_dtype = data_type_dict["fp16"]
        intermediate_dtype = data_type_dict["fp32"]
    elif args.precision in ("fp8", "fp4"):
        qkv_dtype = data_type_dict["fp8"]
        intermediate_dtype = data_type_dict["fp32"]
    output_dtype = get_output_dtype(args.precision, True)
    temp_output_dtype = data_type_dict["fp32"]

    model = FlashAttn(
        qkv_dtype=qkv_dtype,
        intermediate_dtype=intermediate_dtype,
        output_dtype=output_dtype if args.num_splits == 1 else temp_output_dtype,
        is_prefill=args.is_prefill,
        is_causal=args.is_causal,
        num_splits=args.num_splits,
        device=args.device,
    )
    _ = model(
        Tensor([seq_len_q, num_heads_q, head_dim], dtype=qkv_dtype),
        Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
        Tensor([seq_len_kv, num_heads_kv, head_dim], dtype=qkv_dtype),
    )

    latency = (
        model.compile_and_simulate(pcb, drain_l2=(args.num_splits == 1))
        + pcb.compute_module.launch_latency.flashattn
    )
    roofline_latency = model.roofline_model(pcb)

    if args.num_splits > 1:
        model = FlashAttentionCombine(intermediate_dtype, output_dtype)
        _ = model(
            Tensor(
                [seq_len_q, num_heads_q * head_dim, args.num_splits],
                dtype=intermediate_dtype,
            )
        )

        latency += (
            model.compile_and_simulate(pcb)
            + pcb.compute_module.launch_latency.flashattn_combine
        )
        roofline_latency += model.roofline_model(pcb)

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")
