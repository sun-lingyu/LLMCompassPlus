from math import ceil

from software_model.elementwise_like import ElementwiseLike, L2CacheElementwiseLike
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor, data_type_dict


class L2CacheQuant(L2CacheElementwiseLike):
    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        input_dtype: DataType,
        output_dtype: DataType,
        L2Cache_previous: L2Cache = None,
    ):
        super().__init__(l2_size, M, N, input_dtype, output_dtype)
        self.input_scale_tiles = (
            ceil(N / input_dtype.scale_block_size / self.TILE_LENGTH)
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
                    self.n_tiles,
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

        self.inherit(L2Cache_previous)



class Quant(ElementwiseLike):
    """Model dtype conversion with unchanged shape, including FP4 block scales."""

    operator_name = "quant"
    cache_type = L2CacheQuant
    access_types = (
        L2AccessType.ACTIVATION,
        L2AccessType.ACTIVATION_SCALE,
        L2AccessType.OUTPUT,
        L2AccessType.OUTPUT_SCALE,
    )

    def __init__(self, input_dtype: DataType, output_dtype: DataType):
        super().__init__(input_dtype, output_dtype)

    def __call__(self, input: Tensor) -> Tensor:
        assert input.dtype == self.input_dtype
        assert len(input.shape) >= 1
        assert all(dim > 0 for dim in input.shape)

        self.M = 1
        for dim in input.shape[:-1]:
            self.M *= dim
        self.N = input.shape[-1]
        self.io_size = self.output_numel * (
            self.input_dtype.word_size + self.output_dtype.word_size
        )
        # Static FP8 scales have no per-tile traffic; FP4 uses FP8 block scales.
        for dtype in (self.input_dtype, self.output_dtype):
            if dtype.name == "fp4":
                self.io_size += self.output_numel / dtype.scale_block_size
        return Tensor(list(input.shape), self.output_dtype)
