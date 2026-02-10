from math import ceil

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor, data_type_dict


class L2CacheFlashAttnCombine(L2Cache):
    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        num_splits: int,
        activation_dtype: DataType,
        output_dtype: DataType,
        scale_dtype: DataType = data_type_dict["fp8"],  # NVFP4 use fp8 (ue4m3) scale
        L2Cache_previous: L2Cache = None,
    ):
        super().__init__(l2_size)
        # change output to activation
        assert M > 0 and N > 0
        self.num_splits = num_splits
        scale_block_size = output_dtype.scale_block_size

        self.m_tiles = ceil(M / L2Cache.TILE_LENGTH)
        self.n_tiles = ceil(N / L2Cache.TILE_LENGTH)
        self.n_scale_tiles = (
            ceil(N / L2Cache.TILE_LENGTH / scale_block_size)
            if scale_block_size
            else None
        )
        self.activation_tile_size = activation_dtype.word_size * self.TILE_LENGTH**2
        self.output_tile_size = output_dtype.word_size * self.TILE_LENGTH**2
        self.scale_tile_size = scale_dtype.word_size * self.TILE_LENGTH**2

        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.activation_tile_size
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
        width = (
            self.n_tiles * self.num_splits
            if access_type == L2AccessType.ACTIVATION
            else self.n_tiles
            if access_type == L2AccessType.OUTPUT
            else self.n_scale_tiles
            if access_type == L2AccessType.OUTPUT_SCALE
            else None
        )
        tile_size = (
            self.activation_tile_size
            if access_type == L2AccessType.ACTIVATION
            else self.output_tile_size
            if access_type == L2AccessType.OUTPUT
            else self.scale_tile_size
            if access_type == L2AccessType.OUTPUT_SCALE
            else None
        )
        assert height and width and tile_size
        assert (
            coord_tuple[0] % L2Cache.TILE_LENGTH == 0
            and coord_tuple[1] % L2Cache.TILE_LENGTH == 0
        )
        assert (
            scope_tuple[0] % L2Cache.TILE_LENGTH == 0
            and scope_tuple[1] % L2Cache.TILE_LENGTH == 0
        )
        assert (
            coord_tuple[0] >= 0
            and coord_tuple[0] + scope_tuple[0] <= height * L2Cache.TILE_LENGTH
        ), (
            f"coord_tuple[0]: {coord_tuple[0]}, scope_tuple[0]: {scope_tuple[0]}, height * L2Cache.TILE_LENGTH: {height * L2Cache.TILE_LENGTH}"
        )
        assert (
            coord_tuple[1] >= 0
            and coord_tuple[1] + scope_tuple[1] <= width * L2Cache.TILE_LENGTH
        ), (
            f"coord_tuple[1]: {coord_tuple[1]}, scope_tuple[1]: {scope_tuple[1]}, width * L2Cache.TILE_LENGTH: {width * L2Cache.TILE_LENGTH}"
        )

        mem_access_size = 0
        for i in range(
            coord_tuple[0], coord_tuple[0] + scope_tuple[0], L2Cache.TILE_LENGTH
        ):
            for j in range(
                coord_tuple[1], coord_tuple[1] + scope_tuple[1], L2Cache.TILE_LENGTH
            ):
                tile = self.Tile(access_type, (i, j))
                if tile in self.resident_tiles:  # HIT
                    self.resident_tiles.move_to_end(tile)  # update LRU
                else:
                    while self.occupied_size + tile_size > self.l2_size:  # EVICT
                        mem_access_size += self.evict_oldest_tile()
                    if access_type not in (
                        L2AccessType.OUTPUT,
                        L2AccessType.OUTPUT_SCALE,
                    ):  # load from DRAM
                        mem_access_size += tile_size
                    self.occupied_size += tile_size
                    self.resident_tiles[self.Tile(access_type, (i, j))] = None
        self.total_mem_access_size += mem_access_size
        return mem_access_size

    def evict_oldest_tile(self):
        assert self.resident_tiles

        mem_access_size = 0
        oldest_tile = self.resident_tiles.popitem(last=False)[0]
        tile_size = (
            self.activation_tile_size
            if oldest_tile.access_type == L2AccessType.ACTIVATION
            else self.output_tile_size
            if oldest_tile.access_type == L2AccessType.OUTPUT
            else self.scale_tile_size
            if oldest_tile.access_type == L2AccessType.OUTPUT_SCALE
            else None
        )
        if oldest_tile.access_type in (
            L2AccessType.OUTPUT,
            L2AccessType.OUTPUT_SCALE,
        ):
            mem_access_size += tile_size
        self.occupied_size -= tile_size
        self.total_mem_access_size += mem_access_size
        return mem_access_size


class FlashAttentionCombine(Operator):
    def __init__(
        self,
        activation_dtype: DataType,
        output_dtype: DataType,
    ):
        super().__init__()
        self.activation_dtype = activation_dtype
        self.output_dtype = output_dtype

    def __call__(self, input1: Tensor) -> Tensor:
        assert self.activation_dtype == input1.dtype
        assert len(input1.shape) == 3
        self.M = input1.shape[0]
        self.N = input1.shape[1]
        self.num_splits = input1.shape[2]
        self.io_size = (
            self.M
            * self.N
            * (
                self.num_splits * self.activation_dtype.word_size
                + self.output_dtype.word_size
            )
        )
        if self.output_dtype.name == "fp4":
            self.io_size += int(
                data_type_dict["fp8"].word_size
                * self.M
                * self.N
                / self.output_dtype.scale_block_size
            )

        return Tensor((self.M, self.N), self.output_dtype)

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = self.io_size / min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )  # must be io_bound
        return self.roofline_latency

    def compile_and_simulate(
        self, pcb_module: Device, drain_l2: bool = True
    ):  # memory bound operator
        self.l2_status = L2CacheFlashAttnCombine(
            pcb_module.compute_module.l2_size,
            self.M,
            self.N,
            self.num_splits,
            self.activation_dtype,
            self.output_dtype,
        )
        mem_access_size = self.l2_status.access(
            L2AccessType.ACTIVATION, (0, 0), (self.M, self.N * self.num_splits)
        )
        mem_access_size += self.l2_status.access(
            L2AccessType.OUTPUT, (0, 0), (self.M, self.N)
        )
        if self.output_dtype.name == "fp4":
            mem_access_size += self.l2_status.access(
                L2AccessType.OUTPUT_SCALE,
                (0, 0),
                (self.M, self.N / self.output_dtype.scale_block_size),
            )
        if drain_l2:
            mem_access_size += self.l2_status.drain()
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
