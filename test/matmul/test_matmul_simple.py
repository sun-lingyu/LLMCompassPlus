from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse
from design_space_exploration.dse import template_to_system, read_architecture_template

if __name__ == "__main__":
    specs = read_architecture_template("/home/sly/LLMCompass/configs/Orin.json")
    system = template_to_system(specs)
    pcb = system.device

    M = 64
    N = 2560
    K = 2432
    print(f"problem: M {M} N {N} K {K}")

    model = Matmul(data_type=data_type_dict["fp16"])
    _ = model(
        Tensor([M, K]),
        Tensor([K, N]),
    )

    latency =  model.compile_and_simulate(pcb, compile_mode="heuristic-GPU") + 2773 / pcb.compute_module.clock_freq

    # roofline_latency = model.roofline_model(pcb) + 2773 / pcb.compute_module.clock_freq

    print(f"Ours: {latency * 1000} ms")
    # print(f"Roofline: {roofline_latency * 1000} ms")