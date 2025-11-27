import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
# from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil

def read_architecture_template(file_path):
    with open(file_path, "r") as f:
        arch_specs = json.load(f)
    return arch_specs


def template_to_device(arch_specs):
    device_specs = arch_specs["device"]
    compute_chiplet_specs = device_specs["compute_chiplet"]
    io_specs = device_specs["io"]
    core_specs = compute_chiplet_specs["core"]
    sublane_count = core_specs["sublane_count"]
    # vector unit
    vector_unit_specs = core_specs["vector_unit"]
    vector_unit = VectorUnit(vector_unit_specs["throughput"])
    # systolic array
    systolic_array_specs = core_specs["systolic_array"]
    systolic_array = SystolicArray(
        systolic_array_specs["array_height"],
        systolic_array_specs["array_width"],
    )
    # core
    core = Core(
        vector_unit,
        systolic_array,
        sublane_count,
        core_specs["SRAM_KB"] * 1024,
    )
    # compute module
    compute_module = ComputeModule(
        core,
        compute_chiplet_specs["core_count"] * device_specs["compute_chiplet_count"],
        device_specs["frequency_Hz"],
        io_specs["global_buffer_MB"] * 1024 * 1024,
        io_specs["global_buffer_bandwidth_per_cycle_byte"],
        overhead_dict["A100"],
    )
    # io module
    io_module = IOModule(
        io_specs["memory_channel_active_count"]
        * io_specs["pin_count_per_channel"]
        * io_specs["bandwidth_per_pin_bit"]
        // 8,
        io_specs["memory_latency_cycle"],
    )
    # memory module
    memory_module = MemoryModule(
        device_specs["memory"]["total_capacity_GB"] * 1024 * 1024 * 1024
    )
    # device
    device = Device(compute_module, io_module, memory_module)
    
    return device   
