from typing import List
from utils import size
from enum import Enum
from typing import NamedTuple
from collections import OrderedDict

class DataType:
    def __init__(self, name: str, word_size: int, scale_block_size: int = None) -> None:
        self.name = name
        self.word_size = word_size
        self.scale_block_size = scale_block_size

data_type_dict = {"int4": DataType("int4", 0.5), "int8": DataType("int8", 1), "int32": DataType("int32", 4), "fp4": DataType("fp4", 0.5, 16), "fp8": DataType("fp8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}

class Tensor:
    def __init__(
        self, shape: List, data_type
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type

class DeviceType(Enum):
    ORIN = 0
    THOR = 1

class L2AccessType(Enum):
    ACTIVATION = 1
    WEIGHT = 2
    OUTPUT = 3
    ACTIVATION_SCALE = 4 # for fp4
    WEIGHT_SCALE = 5 # for fp4
    OUTPUT_SCALE = 6 # for fp4
    Q = 7 # for flashattn
    K = 8 # for flashattn
    V = 9 # for flashattn
    RESIDUAL_INPUT = 10 # for layernorm
    RESIDUAL_OUTPUT = 11 # for layernorm

class L2Cache:
    TILE_LENGTH = 32
    class Tile(NamedTuple): # squre tile with side length 32
        access_type: L2AccessType
        location_tuple: tuple[int, int]
    
    class Tile3D(NamedTuple):
        access_type: L2AccessType
        location_tuple: tuple[int, int, int]
    
    def __init__(self, l2_size: int):
        assert l2_size > 0
        self.l2_size = l2_size

        self.resident_tiles = OrderedDict() # LRU queue
        self.occupied_size = 0
        self.output_tile_size = None
        self.total_mem_access_size = 0

    def access(self,
               access_type: L2AccessType,
               coord_tuple: tuple[int, int],
               scope_tuple: tuple[int, int]
               ):
        raise NotImplementedError()
    
    def evict_oldest_tile(self):
        raise NotImplementedError()
    
    def drain(self):
        mem_access_size = 0
        while self.resident_tiles:
            oldest_tile = self.resident_tiles.popitem(last=False)[0]
            if oldest_tile.access_type == L2AccessType.OUTPUT:
                mem_access_size += self.output_tile_size
        self.occupied_size = 0
        self.total_mem_access_size += mem_access_size
        return mem_access_size