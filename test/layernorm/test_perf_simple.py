import argparse

from hardware_model.device import device_dict
from software_model.layernorm import FusedLayerNorm
from software_model.utils import Tensor, data_type_dict
from test.layernorm.utils import get_output_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("precision", type=str, choices=["fp16"])
    args = parser.parse_args()
    M = args.M
    N = args.N
    print(f"problem: M {M} N {N}")

    pcb = device_dict[args.device]

    input_dtype = data_type_dict[args.precision]
    output_dtype = get_output_dtype(input_dtype, True)

    model = FusedLayerNorm(input_dtype, output_dtype)
    _ = model(
        Tensor([M, N], input_dtype),
        Tensor([M, N], output_dtype),
    )

    latency = (
        model.compile_and_simulate(pcb) + pcb.compute_module.launch_latency.layernorm
    )

    roofline_latency = model.roofline_model(pcb)

    print(f"Ours: {latency * 1000} ms")
    print(f"Roofline: {roofline_latency * 1000} ms")
