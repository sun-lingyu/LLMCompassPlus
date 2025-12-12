from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse
from design_space_exploration.dse import template_to_system, read_architecture_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("precision", type=str, choices=["fp16", "int8"])
    args = parser.parse_args()
    M = args.M
    N = args.N
    K = args.K
    print(f"problem: M {M} N {N} K {K}")

    specs = read_architecture_template(f"configs/Orin_{args.precision}.json")
    system = template_to_system(specs)
    pcb = system.device

    model = Matmul(data_type=data_type_dict[f"{args.precision}"])
    _ = model(
        Tensor([M, K], data_type_dict[f"{args.precision}"]),
        Tensor([K, N], data_type_dict[f"{args.precision}"]),
    )

    latency =  model.compile_and_simulate(pcb, compile_mode="heuristic-GPU") + 2773 / pcb.compute_module.clock_freq

    # roofline_latency = model.roofline_model(pcb) + 2773 / pcb.compute_module.clock_freq

    print(f"Latency: {latency * 1000} ms")
    # print(f"Roofline: {roofline_latency * 1000} ms")