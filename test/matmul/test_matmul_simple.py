from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    pcb = device_dict["Orin"]

    M = 1024
    N = 1024
    K = 3072
    print(f"problem: M {M} N {N} K {K}")

    model = Matmul(activation_data_type=data_type_dict["fp16"], weight_data_type=data_type_dict["fp16"], intermediate_data_type=data_type_dict["fp32"])
    _ = model(
        Tensor([M, K]),
        Tensor([K, N]),
    )

    latency =  model.compile_and_simulate(pcb, compile_mode="heuristic-GPU") + 2773 / pcb.compute_module.clock_freq

    roofline_latency = model.roofline_model(pcb) + 2773 / pcb.compute_module.clock_freq

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")