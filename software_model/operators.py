from math import ceil

from hardware_model.device import Device


class Operator:
    def __init__(self):
        self.fma_count = 0
        self.io_size = 0

    @staticmethod
    def get_io_cycle_count(mem_access_size: int, pcb_module: Device):
        return ceil(
            mem_access_size
            / (
                pcb_module.io_module.bandwidth
                * pcb_module.io_module.bandwidth_efficiency
                / pcb_module.compute_module.clock_freq
            )
        )
