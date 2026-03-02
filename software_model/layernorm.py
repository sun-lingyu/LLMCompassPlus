from math import ceil

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor
from utils import size


class L2CacheLayerNorm(L2Cache):
    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        dtype: DataType,
        L2Cache_previous: L2Cache = None,
    ):
        super().__init__(l2_size)
        # change output to activation
        assert M > 0 and N > 0
        self.m_tiles = ceil(M / L2Cache.TILE_LENGTH)
        self.n_tiles = ceil(N / L2Cache.TILE_LENGTH)
        self.tile_size = dtype.word_size * self.TILE_LENGTH**2
        self.output_tile_size = self.tile_size

        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.tile_size
            while L2Cache_previous.resident_tiles:
                tile = L2Cache_previous.resident_tiles.popitem(last=False)[0]
                if tile.access_type == L2AccessType.OUTPUT:
                    self.resident_tiles[
                        L2Cache.Tile(L2AccessType.ACTIVATION, tile.coord_tuple)
                    ] = None
                    self.occupied_size += L2Cache_previous.output_tile_size
                assert tile.access_type != L2AccessType.OUTPUT_SCALE

    def access(
        self,
        access_type: L2AccessType,
        coord_tuple: tuple[int, int],
        scope_tuple: tuple[int, int],
    ):
        height = self.m_tiles
        width = self.n_tiles
        tile_size = self.tile_size

        return self._access(
            coord_tuple, scope_tuple, access_type, height, width, tile_size
        )

    def evict_oldest_tile(self):
        assert self.resident_tiles

        oldest_tile = self.resident_tiles.popitem(last=False)[0]
        tile_size = self.tile_size
        self.occupied_size -= tile_size


class FusedLayerNorm(Operator):  # Residual + LayerNorm/RMSNorm
    def __init__(self, dtype: DataType):
        super().__init__()
        self.dtype = dtype

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        assert self.dtype == input1.dtype
        assert input1.dtype == input2.dtype
        assert input1.shape == input2.shape
        self.M = size(input1.shape[:-1])
        self.N = input1.shape[-1]
        self.io_size = self.M * self.N * self.dtype.word_size * 4  # 2 input + 2 output
        return input1, input2

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = self.io_size / min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )  # must be io_bound
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device):  # memory bound operator
        self.l2_status = L2CacheLayerNorm(
            pcb_module.compute_module.l2_size, self.M, self.N, self.dtype
        )
        mem_access_size = self.l2_status.access(
            L2AccessType.ACTIVATION, (0, 0), (self.M, self.N)
        )
        mem_access_size += self.l2_status.access(
            L2AccessType.RESIDUAL_INPUT, (0, 0), (self.M, self.N)
        )
        mem_access_size += self.l2_status.access(
            L2AccessType.RESIDUAL_OUTPUT, (0, 0), (self.M, self.N)
        )
        mem_access_size += self.l2_status.access(
            L2AccessType.OUTPUT, (0, 0), (self.M, self.N)
        )
        mem_access_cycle = ceil(
            mem_access_size
            / (
                pcb_module.io_module.bandwidth
                * pcb_module.io_module.bandwidth_efficiency
                / pcb_module.compute_module.clock_freq
            )
        )
        self.latency = mem_access_cycle / pcb_module.compute_module.clock_freq
        return self.latency
