from collections import OrderedDict
from enum import Enum
from typing import List, NamedTuple


class DataType:
    def __init__(self, name: str, word_size: int, scale_block_size: int = None) -> None:
        self.name = name
        self.word_size = word_size
        self.scale_block_size = scale_block_size


data_type_dict = {
    "int4": DataType("int4", 0.5),
    "int8": DataType("int8", 1),
    "int32": DataType("int32", 4),
    "fp4": DataType("fp4", 0.5, 16),
    "fp8": DataType("fp8", 1),
    "fp16": DataType("fp16", 2),
    "fp32": DataType("fp32", 4),
}


class Tensor:
    def __init__(self, shape: List, dtype) -> None:
        self.shape = shape
        self.dtype = dtype


class DeviceType(Enum):
    ORIN = 0
    THOR = 1


class L2AccessType(Enum):
    ACTIVATION = 1
    WEIGHT = 2
    OUTPUT = 3
    ACTIVATION_SCALE = 4  # for fp4
    WEIGHT_SCALE = 5  # for fp4
    OUTPUT_SCALE = 6  # for fp4
    Q = 7  # for flashattn
    K = 8  # for flashattn
    V = 9  # for flashattn
    RESIDUAL_INPUT = 10  # for layernorm
    RESIDUAL_OUTPUT = 11  # for layernorm


class L2Cache:
    """Physical tile LRU shared across operators; views alias immutable storage IDs."""
    TILE_LENGTH = 16
    _ids = __import__("itertools").count()
    WRITE_TYPES = (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE, L2AccessType.RESIDUAL_OUTPUT)

    class Tile(NamedTuple):
        access_type: L2AccessType
        coord_tuple: tuple[int, int]

    def __init__(self, l2_size: int, write_policy="write_back"):
        assert l2_size > 0
        if write_policy not in ("write_back", "write_through"):
            raise ValueError(write_policy)
        self.l2_size = l2_size
        self.write_policy = write_policy
        self._storage = OrderedDict()  # storage key -> (size bytes, dirty)
        self._aliases = {}             # operator-local Tile -> storage key
        self._role_ids = {role: (next(self._ids), role.name) for role in L2AccessType}
        self.named_tensors = {}
        self.occupied_size = 0
        self.output_tile_size = None
        self.total_mem_access_size = 0
        self.read_bytes = self.writeback_bytes = self.write_through_bytes = 0
        self.l2_access_bytes = 0

    @property
    def resident_tiles(self):
        return OrderedDict((tile, None) for tile, key in self._aliases.items() if key in self._storage)

    @property
    def dirty_tiles(self):
        return {key: size for key, (size, dirty) in self._storage.items() if dirty}

    @property
    def pending_writeback_bytes(self):
        return sum(size for size, dirty in self._storage.values() if dirty)

    def inherit(self, previous, mapper=None):
        """Snapshot full physical cache for independent mapping candidates.
        mapper aliases prior local tiles into the new operator without copying data.
        """
        if previous is None:
            return
        assert previous.l2_size == self.l2_size
        self._storage = previous._storage.copy()
        self.occupied_size = previous.occupied_size
        self.named_tensors = previous.named_tensors.copy()
        if mapper is None:
            mapper = lambda tile: self.Tile(
                L2AccessType.ACTIVATION if tile.access_type == L2AccessType.OUTPUT
                else L2AccessType.ACTIVATION_SCALE, tile.coord_tuple
            ) if tile.access_type in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE) else None
        for tile, key in previous._aliases.items():
            mapped = mapper(tile)
            if mapped is not None:
                self._aliases[mapped] = key

    def alias_role(self, target, source):
        self._role_ids[target] = self._role_ids[source]
        for tile, key in list(self._aliases.items()):
            if tile.access_type == source:
                self._aliases[self.Tile(target, tile.coord_tuple)] = key

    def _key(self, tile):
        key = self._aliases.get(tile)
        if key is None:
            key = (self._role_ids[tile.access_type], tile.coord_tuple)
            self._aliases[tile] = key
        return key

    def access(self, access_type, coord_tuple, scope_tuple):
        raise NotImplementedError()

    def _access(self, coord_tuple, scope_tuple, access_type, height, width, tile_size, *, write=None):
        assert height and width and 0 < tile_size <= self.l2_size
        assert all(x % self.TILE_LENGTH == 0 for x in (*coord_tuple, *scope_tuple))
        assert 0 <= coord_tuple[0] and coord_tuple[0] + scope_tuple[0] <= height * self.TILE_LENGTH
        assert 0 <= coord_tuple[1] and coord_tuple[1] + scope_tuple[1] <= width * self.TILE_LENGTH
        is_write = access_type in self.WRITE_TYPES if write is None else write
        transferred = 0
        for i in range(coord_tuple[0], coord_tuple[0] + scope_tuple[0], self.TILE_LENGTH):
            for j in range(coord_tuple[1], coord_tuple[1] + scope_tuple[1], self.TILE_LENGTH):
                key = self._key(self.Tile(access_type, (i, j)))
                entry = self._storage.get(key)
                if entry is None:
                    while self.occupied_size + tile_size > self.l2_size:
                        transferred += self.evict_oldest_tile()
                    self.occupied_size += tile_size
                    dirty = False
                    if not is_write:
                        transferred += tile_size
                        self.read_bytes += tile_size
                else:
                    assert entry[0] == tile_size, "Aliased storage changed dtype/size"
                    dirty = entry[1]
                    self._storage.move_to_end(key)
                if is_write:
                    if self.write_policy == "write_back":
                        dirty = True
                    else:
                        transferred += tile_size
                        self.write_through_bytes += tile_size
                        dirty = False
                self._storage[key] = (tile_size, dirty)
                self.l2_access_bytes += tile_size
        self.total_mem_access_size += transferred
        return transferred

    def evict_oldest_tile(self):
        _, (size, dirty) = self._storage.popitem(last=False)
        self.occupied_size -= size
        written = size if dirty else 0
        self.writeback_bytes += written
        return written

    def drain(self, *, invalidate=False):
        """Persist dirty tiles exactly once; do not change residency by default."""
        written = self.pending_writeback_bytes
        self._storage = OrderedDict((key, (size, False)) for key, (size, _) in self._storage.items())
        self.writeback_bytes += written
        self.total_mem_access_size += written
        if invalidate:
            self._storage.clear()
            self.occupied_size = 0
        return written

    def metrics(self):
        return dict(read_bytes=self.read_bytes, evicted_writeback_bytes=self.writeback_bytes,
                    write_through_bytes=self.write_through_bytes,
                    pending_dirty_bytes=self.pending_writeback_bytes,
                    resident_bytes=self.occupied_size, dram_bytes=self.total_mem_access_size)
