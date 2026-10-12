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

    def finish_cache_accounting(self, pcb_module, drain_at_end):
        self.execution_cycles = self.latency * pcb_module.compute_module.clock_freq
        self.cache_metrics = self.l2_status.metrics()
        self.drain_bytes = self.l2_status.drain() if drain_at_end else 0
        self.drain_cycles = self.get_io_cycle_count(self.drain_bytes, pcb_module)
        self.latency = (self.execution_cycles + self.drain_cycles) / pcb_module.compute_module.clock_freq
        self.mem_access_size = self.l2_status.total_mem_access_size
        self.cache_metrics.update(drain_bytes=self.drain_bytes, drain_cycles=self.drain_cycles,
                                  pending_dirty_after_drain=self.l2_status.pending_writeback_bytes,
                                  total_dram_bytes=self.mem_access_size)
        if drain_at_end:
            print(f"Cache accounting: execution_cycles={self.execution_cycles:.3f}, drain_cycles={self.drain_cycles}, drain_bytes={self.drain_bytes}, eviction_writeback_bytes={self.cache_metrics['evicted_writeback_bytes']}")
        return self.latency
