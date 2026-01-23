from utils import size
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType, L2AccessType, L2Cache, data_type_dict
from math import ceil

class L2CacheLayerNorm(L2Cache):
    def __init__(self,
                l2_size: int,
                M: int,
                N: int,
                dtype: DataType,
                L2Cache_previous: L2Cache = None
                ):
        super().__init__(l2_size)
        # change output to activation
        assert M > 0 and N > 0
        self.m_tiles = ceil(M / L2Cache.TILE_LENGTH)
        self.n_tiles = ceil(N / L2Cache.TILE_LENGTH)
        self.tile_size = dtype.word_size * self.TILE_LENGTH ** 2
        self.output_tile_size = self.tile_size

        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.activation_tile_size
            while L2Cache_previous.resident_tiles:
                tile = L2Cache_previous.resident_tiles.popitem(last=False)[0]
                if tile.access_type == L2AccessType.OUTPUT:
                    self.resident_tiles[L2Cache.Tile(L2AccessType.ACTIVATION, tile.coord_tuple)] = None
                    self.occupied_size += L2Cache_previous.output_tile_size
                assert tile.access_type != L2AccessType.OUTPUT_SCALE
    
    def access(self,
                access_type: L2AccessType,
                coord_tuple: tuple[int, int],
                scope_tuple: tuple[int, int]
                ):
        height = self.m_tiles
        width = self.n_tiles
        tile_size = self.tile_size
        assert height and width
        assert coord_tuple[0] % L2Cache.TILE_LENGTH == 0 and coord_tuple[1] % L2Cache.TILE_LENGTH == 0
        assert scope_tuple[0] % L2Cache.TILE_LENGTH == 0 and scope_tuple[1] % L2Cache.TILE_LENGTH == 0
        assert coord_tuple[0] >= 0 and coord_tuple[0] + scope_tuple[0] <= height * L2Cache.TILE_LENGTH, f"coord_tuple[0]: {coord_tuple[0]}, scope_tuple[0]: {scope_tuple[0]}, height * L2Cache.TILE_LENGTH: {height * L2Cache.TILE_LENGTH}"
        assert coord_tuple[1] >= 0 and coord_tuple[1] + scope_tuple[1] <= width * L2Cache.TILE_LENGTH, f"coord_tuple[1]: {coord_tuple[1]}, scope_tuple[1]: {scope_tuple[1]}, width * L2Cache.TILE_LENGTH: {width * L2Cache.TILE_LENGTH}"

        mem_access_size = 0
        for i in range(coord_tuple[0], coord_tuple[0] + scope_tuple[0], L2Cache.TILE_LENGTH):
            for j in range(coord_tuple[1], coord_tuple[1] + scope_tuple[1], L2Cache.TILE_LENGTH):
                tile = self.Tile(access_type, (i, j))
                if tile in self.resident_tiles: # HIT
                    self.resident_tiles.move_to_end(tile) # update LRU
                else:
                    while self.occupied_size + tile_size > self.l2_size: # EVICT
                        mem_access_size += self.evict_oldest_tile()
                    if access_type not in (L2AccessType.OUTPUT, L2AccessType.RESIDUAL_OUTPUT): # load from DRAM
                        mem_access_size += tile_size
                    self.occupied_size += tile_size
                    self.resident_tiles[self.Tile(access_type, (i, j))] = None
        self.total_mem_access_size += mem_access_size
        return mem_access_size

    def evict_oldest_tile(self):
        assert self.resident_tiles
        
        mem_access_size = 0
        oldest_tile = self.resident_tiles.popitem(last=False)[0]
        tile_size = self.tile_size
        if oldest_tile.access_type in (L2AccessType.OUTPUT, L2AccessType.RESIDUAL_OUTPUT):
            mem_access_size += tile_size
        self.occupied_size -= tile_size
        self.total_mem_access_size += mem_access_size
        return mem_access_size
    
    def drain(self):
        mem_access_size = 0
        while self.resident_tiles:
            oldest_tile = self.resident_tiles.popitem(last=False)[0]
            if oldest_tile.access_type in (L2AccessType.RESIDUAL_OUTPUT, L2AccessType.OUTPUT):
                mem_access_size += self.output_tile_size
        self.occupied_size = 0
        self.total_mem_access_size += mem_access_size
        return mem_access_size

class FusedLayerNorm(Operator): # Residual + LayerNorm/RMSNorm
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        assert self.data_type == input1.data_type
        assert input1.data_type == input2.data_type
        assert input1.shape == input2.shape
        self.shape = input1.shape
        self.M = size(input1.shape[:-1])
        self.N = input1.shape[-1]
        self.io_count = self.M * self.N * self.data_type.word_size * 4 # 2 input + 2 output
        self.fma_count = self.M * self.N
        return input1, input2

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = self.io_count / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ) # must be io_bound
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device): # memory bound operator
        self.l2_status = L2CacheLayerNorm(pcb_module.compute_module.l2_size, self.M, self.N, self.data_type)
        mem_access_size = self.l2_status.access(L2AccessType.ACTIVATION, (0, 0), (self.M, self.N))
        mem_access_size += self.l2_status.access(L2AccessType.RESIDUAL_INPUT, (0, 0), (self.M, self.N))
        mem_access_size += self.l2_status.access(L2AccessType.RESIDUAL_OUTPUT, (0, 0), (self.M, self.N))
        mem_access_size += self.l2_status.access(L2AccessType.OUTPUT, (0, 0), (self.M, self.N))
        mem_access_size += self.l2_status.drain()
        mem_access_cycle = ceil(
            mem_access_size
            / (
                pcb_module.io_module.bandwidth
                * pcb_module.io_module.bandwidth_efficiency
                / pcb_module.compute_module.clock_freq
            ))
        self.latency = mem_access_cycle / pcb_module.compute_module.clock_freq
        return self.latency
