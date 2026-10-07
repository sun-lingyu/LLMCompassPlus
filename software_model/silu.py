from math import ceil

from software_model.elementwise_like import ElementwiseLike, L2CacheElementwiseLike
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor, data_type_dict


class L2CacheSiLU(L2CacheElementwiseLike):
    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        input_dtype: DataType,
        output_dtype: DataType,
        L2Cache_previous: L2Cache = None,
        input_N: int = None,
    ):
        super().__init__(l2_size, M, N, input_dtype, output_dtype)
        self.input_n_tiles = ceil((input_N or N) / self.TILE_LENGTH)
        self.input_scale_tiles = (
            ceil((input_N or N) / input_dtype.scale_block_size / self.TILE_LENGTH)
            if input_dtype.name == "fp4"
            else 0
        )
        self.output_scale_tiles = (
            ceil(N / output_dtype.scale_block_size / self.TILE_LENGTH)
            if output_dtype.name == "fp4"
            else 0
        )
        self.scale_tile_size = data_type_dict["fp8"].word_size * self.TILE_LENGTH**2
        self.regions.update(
            {
                L2AccessType.ACTIVATION: (
                    self.m_tiles,
                    self.input_n_tiles,
                    self.input_tile_size,
                ),
                L2AccessType.ACTIVATION_SCALE: (
                    self.m_tiles,
                    self.input_scale_tiles,
                    self.scale_tile_size,
                ),
                L2AccessType.OUTPUT_SCALE: (
                    self.m_tiles,
                    self.output_scale_tiles,
                    self.scale_tile_size,
                ),
            }
        )

        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.input_tile_size
            for tile in L2Cache_previous.resident_tiles.keys():
                if tile.access_type == L2AccessType.OUTPUT:
                    self.resident_tiles[
                        L2Cache.Tile(L2AccessType.ACTIVATION, tile.coord_tuple)
                    ] = None
                    self.occupied_size += L2Cache_previous.output_tile_size
                if tile.access_type == L2AccessType.OUTPUT_SCALE:
                    assert self.input_scale_tiles
                    self.resident_tiles[
                        L2Cache.Tile(L2AccessType.ACTIVATION_SCALE, tile.coord_tuple)
                    ] = None
                    self.occupied_size += self.scale_tile_size


class SiLU(ElementwiseLike):
    """Elementwise SiLU, optionally fused with multiplication by the up branch."""

    operator_name = "silu"
    cache_type = L2CacheSiLU
    access_types = (
        L2AccessType.ACTIVATION,
        L2AccessType.ACTIVATION_SCALE,
        L2AccessType.OUTPUT,
        L2AccessType.OUTPUT_SCALE,
    )

    def __init__(
        self,
        input_dtype: DataType,
        output_dtype: DataType = None,
        fused_mul: bool = False,
    ):
        super().__init__(input_dtype, output_dtype or input_dtype)
        self.fused_mul = fused_mul

    def __call__(self, input: Tensor) -> Tensor:
        assert input.dtype == self.input_dtype
        assert len(input.shape) >= 1
        assert all(dim > 0 for dim in input.shape)

        self.M = 1
        for dim in input.shape[:-1]:
            self.M *= dim
        self.input_N = input.shape[-1]
        assert not self.fused_mul or self.input_N % 2 == 0
        self.N = self.input_N // 2 if self.fused_mul else self.input_N
        input_size = self.M * self.input_N * self.input_dtype.word_size
        output_size = self.M * self.N * self.output_dtype.word_size
        self.io_size = input_size + output_size

        # Static FP8 scales add no per-tile tensor traffic. FP4 retains block scales.
        if self.input_dtype.name == "fp4":
            self.io_size += self.M * self.input_N / self.input_dtype.scale_block_size
        if self.output_dtype.name == "fp4":
            self.io_size += self.M * self.N / self.output_dtype.scale_block_size
        return Tensor(list(input.shape[:-1]) + [self.N], self.output_dtype)

    def _cache_kwargs(self):
        return {"input_N": self.input_N}
