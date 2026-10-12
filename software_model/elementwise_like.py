"""Shared latency and tiled-cache models for elementwise-like operators."""

from math import ceil

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, L2AccessType, L2Cache


class L2CacheElementwiseLike(L2Cache):
    """Regions map each access type to (tile rows, tile columns, bytes per tile)."""

    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        input_dtype: DataType,
        output_dtype: DataType,
    ):
        super().__init__(l2_size)
        assert M > 0 and N > 0
        self.M = M
        self.N = N
        self.m_tiles = ceil(M / self.TILE_LENGTH)
        self.n_tiles = ceil(N / self.TILE_LENGTH)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.input_tile_size = input_dtype.word_size * self.TILE_LENGTH**2
        self.output_tile_size = output_dtype.word_size * self.TILE_LENGTH**2
        self.regions = {
            L2AccessType.ACTIVATION: (self.m_tiles, self.n_tiles, self.input_tile_size),
            L2AccessType.OUTPUT: (self.m_tiles, self.n_tiles, self.output_tile_size),
        }

    def access(self, access_type, coord_tuple, scope_tuple):
        height, width, tile_size = self.regions[access_type]
        return self._access(
            coord_tuple, scope_tuple, access_type, height, width, tile_size
        )

    def _tile_size(self, access_type):
        return self.regions[access_type][2]



class ElementwiseLike(Operator):
    """Overlap tiled memory traffic with seconds-per-logical-output-element work."""

    operator_name = None
    cache_type = None
    access_types = (L2AccessType.ACTIVATION, L2AccessType.OUTPUT)

    def __init__(self, input_dtype: DataType, output_dtype: DataType):
        super().__init__()
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    @property
    def output_numel(self):
        # Subclasses set N to the processed output width: no residual or V copy.
        return self.M * self.N

    def _cache_kwargs(self):
        return {}

    def _create_l2_cache(self, pcb_module, L2Cache_previous):
        return self.cache_type(
            pcb_module.compute_module.l2_size,
            self.M,
            self.N,
            self.input_dtype,
            self.output_dtype,
            L2Cache_previous=L2Cache_previous,
            **self._cache_kwargs(),
        )

    def _iter_accesses(self):
        for access_type in self.access_types:
            height, width, _ = self.l2_status.regions[access_type]
            if height and width:
                yield (
                    access_type,
                    (0, 0),
                    (height * L2Cache.TILE_LENGTH, width * L2Cache.TILE_LENGTH),
                )

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = self.io_size / min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )
        return self.roofline_latency

    def compile_and_simulate(
        self, pcb_module: Device, L2Cache_previous: L2Cache = None, *, drain_at_end: bool = False
    ):
        self.l2_status = self._create_l2_cache(pcb_module, L2Cache_previous)
        self.mem_access_size = sum(
            self.l2_status.access(access_type, coord, scope)
            for access_type, coord, scope in self._iter_accesses()
        )
        self.memory_latency = (
            max(self.get_io_cycle_count(self.mem_access_size, pcb_module),
                self.l2_status.l2_access_bytes / pcb_module.compute_module.l2_bandwidth_per_cycle)
            / pcb_module.compute_module.clock_freq
        )
        factor = pcb_module.compute_module.compute_latency_factor[self.operator_name]
        self.compute_latency = self.output_numel * factor
        self.latency = max(self.memory_latency, self.compute_latency)
        return self.finish_cache_accounting(pcb_module, drain_at_end)
