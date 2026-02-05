import argparse

from hardware_model.device import device_dict
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
from test.matmul.utils import get_output_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument(
        "op_name", type=str, choices=["qkv_proj", "o_proj", "up_proj", "down_proj"]
    )
    parser.add_argument(
        "precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()
    M = args.M
    N = args.N
    K = args.K
    print(f"problem: M {M} N {N} K {K}")

    pcb = device_dict[args.device]

    if args.precision == "fp16":
        activation_dtype = data_type_dict["fp16"]
        weight_dtype = data_type_dict["fp16"]
        intermediate_dtype = data_type_dict["fp32"]
    elif args.precision == "int8":
        activation_dtype = data_type_dict["int8"]
        weight_dtype = data_type_dict["int8"]
        intermediate_dtype = data_type_dict["int32"]
    elif args.precision == "int4":
        activation_dtype = data_type_dict["fp16"]
        weight_dtype = data_type_dict["int4"]
        intermediate_dtype = data_type_dict["fp32"]
    elif args.precision == "fp8":
        activation_dtype = data_type_dict["fp8"]
        weight_dtype = data_type_dict["fp8"]
        intermediate_dtype = data_type_dict["fp32"]
    elif args.precision == "fp4":
        activation_dtype = data_type_dict["fp4"]
        weight_dtype = data_type_dict["fp4"]
        intermediate_dtype = data_type_dict["fp32"]
    output_dtype = get_output_dtype(activation_dtype, args.op_name, True)

    model = Matmul(
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        intermediate_dtype=intermediate_dtype,
        output_dtype=output_dtype,
        device=args.device,
    )
    _ = model(
        Tensor([M, K], dtype=activation_dtype),
        Tensor([K, N], dtype=weight_dtype),
    )

    latency = model.compile_and_simulate(pcb) + pcb.compute_module.launch_latency.matmul

    roofline_latency = model.roofline_model(pcb)

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")
