from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from design_space_exploration.dse import template_to_system, read_architecture_template
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--roofline", action="store_true", help="Roofline simulation")
    args = parser.parse_args()

    if args.simgpu:
        pcb = device_dict["Orin"]

    K = 12288
    N = K
    titile = f"Performance of Matmul with K={K}, N={N}"
    print(f"{titile}")

    test_overhead = True

    for M in range(5, 16):
        M = 2**M
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        if args.gpu:
            if test_overhead:
                model.gpu_kernel_launch_overhead()
                test_overhead = False
            latency = model.run_on_gpu()
        if args.simgpu:
            if args.roofline:
                latency = model.roofline_model(pcb) + 2.1e-5
                file_name='matmul_A100_roofline.csv'
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-GPU")
                    + 2.1e-5
                )
                file_name='matmul_A100_sim.csv'
        tflops = 2 * M * N * K / latency / 1e12
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops", flush=True)
        with open(f'test/matmul/{file_name}', 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops\n")

    M = 8192
    print(f"Performance of Matmul with M={M}, N=K")
    for K in range(5, 16):
        K = 2**K
        N = K
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        if args.gpu:
            latency = model.run_on_gpu()
        if args.simgpu:
            if args.roofline:
                latency = model.roofline_model(pcb) + 2.1e-5
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-GPU")
                    + 2.1e-5
                )
        tflops = 2 * M * N * K / latency / 1e12
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops", flush=True)
        with open(f'test/matmul/{file_name}', 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops\n")
