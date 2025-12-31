from hardware_model.compute_module import ComputeModule, compute_module_dict
from hardware_model.io_module import IOModule, IO_module_dict
from hardware_model.memory_module import MemoryModule, memory_module_dict


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


device_dict = {
    "Thor": Device(
        compute_module_dict["Thor"],
        IO_module_dict["Thor"],
        memory_module_dict["Thor"],
    ),
    "Orin": Device(
        compute_module_dict["Orin"],
        IO_module_dict["Orin"],
        memory_module_dict["Orin"],
    ),
    "A100": Device(
        compute_module_dict["A100"],
        IO_module_dict["A100"],
        memory_module_dict["A100"],
    ),
}
