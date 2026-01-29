import glob
import json
import os

from hardware_model.compute_module import (
    ComputeModule,
    Core,
    LaunchLatency,
    SystolicArray,
    VectorUnit,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule


class Device:
    def __init__(
        self,
        compute_module: ComputeModule,
        io_module: IOModule,
        memory_module: MemoryModule,
    ) -> None:
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module


def _create_device_from_config(config_data: dict) -> Device:
    """
    Helper function to instantiate a Device object from a dictionary.
    """
    # 1. Build Compute Module Components
    comp_params = config_data["compute_module"]
    core_params = comp_params["core"]

    vector_unit = VectorUnit(**core_params["vector_unit"])
    systolic_array = SystolicArray(**core_params["systolic_array"])

    core = Core(
        vector_unit=vector_unit,
        systolic_array=systolic_array,
        total_registers=core_params["total_registers"],
        sublane_count=core_params["sublane_count"],
        SRAM_size=core_params["SRAM_size"],
    )

    launch_latency = LaunchLatency(**comp_params["launch_latency"])

    compute_module = ComputeModule(
        core=core,
        core_count=comp_params["core_count"],
        clock_freq=comp_params["clock_freq"],
        l2_size=comp_params["l2_size"],
        l2_bandwidth_per_cycle=comp_params["l2_bandwidth_per_cycle"],
        l2_latency_cycles=comp_params["l2_latency_cycles"],
        launch_latency=launch_latency,
    )

    # 2. Build IO Module
    io_module = IOModule(**config_data["io_module"])

    # 3. Build Memory Module
    memory_module = MemoryModule(**config_data["memory_module"])

    return Device(compute_module, io_module, memory_module)


def _load_all_devices(config_dir: str):
    """
    Scans the config directory for .json files and loads them.
    Device name is derived from the filename.
    """
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    devices = {}
    json_files = glob.glob(os.path.join(config_dir, "*.json"))

    for file_path in json_files:
        # Extract name from filename (e.g., 'Thor.json' -> 'Thor')
        device_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            devices[device_name] = _create_device_from_config(data)

    return devices


# Define path to the 'configs' folder relative to this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_config_dir = os.path.join(_current_dir, "configs")

# Expose the dictionary
device_dict = _load_all_devices(_config_dir)
