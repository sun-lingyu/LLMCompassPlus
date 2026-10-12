from software_model.elementwise_like import ElementwiseLike, L2CacheElementwiseLike
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor


class L2CacheQKNormRoPE(L2CacheElementwiseLike):
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

        self.inherit(L2Cache_previous, lambda tile: tile if tile.access_type == L2AccessType.OUTPUT else None)
        self.alias_role(L2AccessType.ACTIVATION, L2AccessType.OUTPUT)


class FusedQKNormRoPE(ElementwiseLike):
    operator_name = "rope"
    cache_type = L2CacheQKNormRoPE

    def __init__(
        self,
        input_dtype: DataType,
        output_dtype: DataType = None,
    ):
        super().__init__(input_dtype, output_dtype or input_dtype)
        assert self.input_dtype == self.output_dtype

    def __call__(
        self,
        qkv: Tensor,
        head_dim,
        num_heads_q,
        num_heads_kv,
    ) -> Tensor:
        assert qkv.dtype == self.input_dtype
        assert len(qkv.shape) >= 2
        assert num_heads_q > 0 and num_heads_kv > 0
        assert head_dim > 0

        total_heads = num_heads_q + num_heads_kv * 2
        assert qkv.shape[-1] == total_heads * head_dim

        self.M = 1
        for dim in qkv.shape[:-1]:
            assert dim > 0
            self.M *= dim

        self.N = (num_heads_q + num_heads_kv) * head_dim
        input_size = self.M * self.N * self.input_dtype.word_size
        output_size = self.M * self.N * self.output_dtype.word_size
        self.io_size = input_size + output_size

        # The operation is in-place; the packed V region is returned unchanged.
        return qkv
