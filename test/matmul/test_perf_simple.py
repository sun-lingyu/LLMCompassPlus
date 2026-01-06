from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=["Orin", "Thor"],)
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"])
    args = parser.parse_args()
    M = args.M
    N = args.N
    K = args.K
    print(f"problem: M {M} N {N} K {K}")

    pcb = device_dict[args.device]

    if args.precision == "fp16":
        activation_data_type=data_type_dict["fp16"]
        weight_data_type=data_type_dict["fp16"]
        intermediate_data_type=data_type_dict["fp32"]
    elif args.precision == "int8":
        activation_data_type=data_type_dict["int8"]
        weight_data_type=data_type_dict["int8"]
        intermediate_data_type=data_type_dict["int32"]
    elif args.precision == "int4":
        activation_data_type=data_type_dict["fp16"]
        weight_data_type=data_type_dict["int4"]
        intermediate_data_type=data_type_dict["fp32"]
    elif args.precision == "fp8":
        activation_data_type=data_type_dict["fp8"]
        weight_data_type=data_type_dict["fp8"]
        intermediate_data_type=data_type_dict["fp32"]
    elif args.precision == "fp4":
        activation_data_type=data_type_dict["fp4"]
        weight_data_type=data_type_dict["fp4"]
        intermediate_data_type=data_type_dict["fp32"]

    model = Matmul(activation_data_type=activation_data_type, weight_data_type=weight_data_type, intermediate_data_type=intermediate_data_type, device=args.device)
    _ = model(
        Tensor([M, K], data_type=activation_data_type),
        Tensor([K, N], data_type=weight_data_type),
    )

    latency =  model.compile_and_simulate(pcb, compile_mode="heuristic-GPU")

    roofline_latency = model.roofline_model(pcb)

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")