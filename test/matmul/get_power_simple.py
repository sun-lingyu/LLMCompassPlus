import os
import argparse
from test.matmul.get_power import measure_power_remote
file_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("op_name", type=str)
    parser.add_argument("device", type=str)
    parser.add_argument("precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"])
    args = parser.parse_args()

    p1, p2 = measure_power_remote(args.m, args.n, args.k, args.op_name, args.precision, device=args.device, ignore_cache=True, total_duration=10)
    print(f"M N K {args.m} {args.n} {args.k} precision {args.precision} Power VDD_GPU_SOC {p1:.2f}W Power VDDQ_VDD2_1V8AO {p2:.2f}W")
