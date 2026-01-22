from software_model.layernorm import FusedLayerNorm
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=["Orin", "Thor"],)
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("precision", type=str, choices=["fp16"])
    args = parser.parse_args()
    M = args.M
    N = args.N
    print(f"problem: M {M} N {N}")

    pcb = device_dict[args.device]

    model = FusedLayerNorm(data_type_dict["fp16"])
    _ = model(
        Tensor([M, N], data_type=data_type_dict["fp16"]),
        Tensor([M, N], data_type=data_type_dict["fp16"]),
    )

    latency = max(model.compile_and_simulate(pcb), pcb.compute_module.launch_latency.layernorm)

    roofline_latency = model.roofline_model(pcb)

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")