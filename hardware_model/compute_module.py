from math import ceil
from software_model.utils import DataType, data_type_dict


class VectorUnit:
    def __init__(
        self,
        throughput: dict
    ):
        self.throughput = throughput

    def get_throughput_per_cycle(self, data_type: DataType, operation: str):
        data_type = data_type.name
        assert data_type in ["int32", "fp16", "fp32", "fp64", "int8"], f"Datatype {data_type} not supported in VectorUnit"
        assert operation in ["exp2", "cvt", "reduction", "fma"], f"Operation {operation} not supported in VectorUnit"
        if operation == "exp2":
            return self.throughput["exp2"]
        if operation == "cvt" and (data_type == "int32" or data_type == "fp32"):
            return self.throughput["cvt_int32_fp32"]
        if operation == "cvt" and data_type == "fp16":
            return self.throughput["cvt_int32_fp32"]
        if operation == "cvt" and data_type == "int8":
            return self.throughput["cvt_int32_int8"]
        if operation == "reduction" and data_type == "int32": # special case for int32 add/sub
            return self.throughput["int32"] * 2
        else:
            return self.throughput[data_type]

vector_unit_dict = {
    "Orin": VectorUnit({"int32": 16,
                        "fp16": 64,
                        "fp32": 32,
                        "fp64": 0.5,
                        "exp2": 4,
                        "cvt_int32_fp32": 16,
                        "cvt_fp32_fp16": 16,
                        "cvt_int32_int8": 16
                    }),
    "A100": VectorUnit({"int32": 16,
                        "fp16": 64,
                        "fp32": 16,
                        "fp64": 8,
                        "exp2": 4,
                        "cvt_int32_fp32": 4,
                        "cvt_fp32_fp16": 16
                    }),
}

class SystolicArray:
    def __init__(
        self,
        array_width,
        array_height,
    ):
        self.array_width = array_width
        self.array_height = array_height

systolic_array_dict = {
    "Orin": SystolicArray(16, 8),
    "A100": SystolicArray(16, 8),
}

class Core:
    def __init__(
        self,
        vector_unit: VectorUnit,
        systolic_array: SystolicArray,
        total_registers: int,
        sublane_count,
        SRAM_size,
    ):
        self.vector_unit = vector_unit
        self.systolic_array = systolic_array
        self.total_registers = total_registers
        self.sublane_count = sublane_count
        self.SRAM_size = SRAM_size  # Byte

core_dict = {
    "SM_Orin": Core(
        vector_unit_dict["Orin"], systolic_array_dict["Orin"], 65536, 4, 192 * 1024
    ),
    "SM_A100": Core(
        vector_unit_dict["A100"], systolic_array_dict["A100"], 65536, 4, 192 * 1024
    ),
}
# compute_tile_dict={'SM_A100_int8':ComputeTile(512, 4096, 192*1024*8,3.41, 'TSMC N7', 128*8),'SM_A100_fp16':ComputeTile(512, 2048, 192*1024*8,3.41, 'TSMC N7', 128),}
# flops: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch__fig2
# area: https://pbs.twimg.com/media/FOT_-NJWUAARrtB?format=jpg&name=large

class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu

overhead_dict = {
    "Orin": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "A100": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
}

class ComputeModule:
    def __init__(
        self,
        core: Core,
        core_count,
        clock_freq,
        l2_size,
        l2_bandwidth_per_cycle,
        l2_latency_cycles,
        overhead: Overhead = overhead_dict["A100"],
    ):
        self.core = core
        self.core_count = core_count
        self.clock_freq = clock_freq
        self.l2_size = int(l2_size)  # Byte
        self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle  # Byte/clock
        self.l2_latency_cycles = l2_latency_cycles
        self.overhead = overhead
    
    def get_total_vector_throughput_per_cycle(self, data_type: DataType, operation: str):
        return self.core.vector_unit.get_throughput_per_cycle(data_type, operation) * self.core.sublane_count * self.core_count
    
    def get_total_systolic_array_throughput_per_cycle(self, data_type: DataType):
        return self.core_count * self.core.sublane_count * self.core.systolic_array.array_height * self.core.systolic_array.array_width * (4 / data_type.word_size)


compute_module_dict = {
    "Orin": ComputeModule(
        core_dict["SM_Orin"],
        16,
        1301e6,
        4 * 1024**2,
        512,
        146,
        overhead_dict["Orin"],
    ),
    "A100": ComputeModule(
        core_dict["SM_A100"],
        108,
        1410e6,
        40 * 1024**2,
        5120,
        223,
        overhead_dict["A100"],
    ),
}
