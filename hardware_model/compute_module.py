from math import ceil
from software_model.utils import DataType


class VectorUnit:
    def __init__(self, throughput: dict):
        self.throughput = throughput

    def get_throughput_per_cycle(self, data_type: DataType, operation: str):
        data_type_name = data_type.name
        # Validation
        valid_types = ["int32", "fp16", "fp32", "fp64", "int8", "fp8", "fp4"]
        valid_ops = ["exp2", "cvt", "reduction", "fma"]
        
        assert data_type_name in valid_types, f"Datatype {data_type_name} not supported"
        assert operation in valid_ops, f"Operation {operation} not supported"

        if operation == "exp2":
            return self.throughput["exp2"]
        if operation == "cvt":
            if data_type_name in ["int32", "fp32"]:
                return self.throughput["cvt_int32_fp32"]
            if data_type_name == "fp16":
                return self.throughput["cvt_fp32_fp16"]
            if data_type_name == "int8":
                return self.throughput["cvt_int32_int8"]
        if operation == "reduction" and data_type_name == "int32":
            return self.throughput["int32"] * 2
        
        return self.throughput[data_type_name]


class SystolicArray:
    def __init__(self, array_width, array_height):
        self.array_width = array_width
        self.array_height = array_height


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


class LaunchLatency:
    def __init__(self, matmul, layernorm, flashattn):
        self.matmul = matmul
        self.layernorm = layernorm
        self.flashattn = flashattn


class ComputeModule:
    def __init__(
        self,
        core: Core,
        core_count,
        clock_freq,
        l2_size,
        l2_bandwidth_per_cycle,
        l2_latency_cycles,
        launch_latency: LaunchLatency,
    ):
        self.core = core
        self.core_count = core_count
        self.clock_freq = clock_freq
        self.l2_size = int(l2_size)  # Byte
        self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle  # Byte/clock
        self.l2_latency_cycles = l2_latency_cycles
        self.launch_latency = launch_latency
    
    def get_total_vector_throughput_per_cycle(self, data_type: DataType, operation: str):
        return (self.core.vector_unit.get_throughput_per_cycle(data_type, operation) 
                * self.core.sublane_count 
                * self.core_count)
    
    def get_total_systolic_array_throughput_per_cycle(self, data_type: DataType):
        return (self.core_count 
                * self.core.sublane_count 
                * self.core.systolic_array.array_height 
                * self.core.systolic_array.array_width 
                * (4 // data_type.word_size))