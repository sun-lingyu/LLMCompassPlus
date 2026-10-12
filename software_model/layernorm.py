from software_model.elementwise_like import ElementwiseLike, L2CacheElementwiseLike
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor


class L2CacheLayerNorm(L2CacheElementwiseLike):
    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        input_dtype: DataType,
        output_dtype: DataType,
        residual_dtype: DataType,
        L2Cache_previous: L2Cache = None,
    ):
        super().__init__(l2_size, M, N, input_dtype, output_dtype)
        assert residual_dtype is not None
        self.residual_tile_size = residual_dtype.word_size * self.TILE_LENGTH**2
        self.regions.update(
            {
                L2AccessType.RESIDUAL_INPUT: (
                    self.m_tiles,
                    self.n_tiles,
                    self.input_tile_size,
                ),
                L2AccessType.RESIDUAL_OUTPUT: (
                    self.m_tiles,
                    self.n_tiles,
                    self.residual_tile_size,
                ),
            }
        )

        self.inherit(L2Cache_previous)
        self._role_ids[L2AccessType.RESIDUAL_INPUT] = self.named_tensors.get("residual", self._role_ids[L2AccessType.RESIDUAL_INPUT])
        self.named_tensors["residual"] = self._role_ids[L2AccessType.RESIDUAL_OUTPUT]



class FusedLayerNorm(ElementwiseLike):  # Residual + LayerNorm/RMSNorm
    operator_name = "layernorm"
    cache_type = L2CacheLayerNorm
    access_types = (
        L2AccessType.ACTIVATION,
        L2AccessType.RESIDUAL_INPUT,
        L2AccessType.RESIDUAL_OUTPUT,
        L2AccessType.OUTPUT,
    )

    def __init__(
        self, input_dtype: DataType, output_dtype: DataType, residual_dtype: DataType
    ):
        super().__init__(input_dtype, output_dtype)
        assert residual_dtype is not None
        assert residual_dtype == input_dtype
        self.residual_dtype = residual_dtype

    def __call__(self, input1: Tensor, input2: Tensor) -> tuple[Tensor, Tensor]:
        assert self.input_dtype == input1.dtype
        assert input1.dtype == input2.dtype
        assert input1.shape == input2.shape
        self.M = 1
        for dim in input1.shape[:-1]:
            self.M *= dim
        self.N = input1.shape[-1]
        self.io_size = (
            self.M
            * self.N
            * (
                2 * self.input_dtype.word_size
                + self.output_dtype.word_size
                + self.residual_dtype.word_size
            )
        )
        return (
            Tensor(list(input1.shape), self.output_dtype),
            Tensor(list(input2.shape), self.residual_dtype),
        )

    def _cache_kwargs(self):
        return {"residual_dtype": self.residual_dtype}
