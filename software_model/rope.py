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

        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.input_tile_size
            for tile in L2Cache_previous.resident_tiles.keys():
                if tile.access_type == L2AccessType.OUTPUT:
                    self.resident_tiles[
                        L2Cache.Tile(
                            L2AccessType.ACTIVATION
                            if tile.coord_tuple[1] < N
                            else L2AccessType.OUTPUT,
                            tile.coord_tuple,
                        )
                    ] = None
                    self.occupied_size += L2Cache_previous.output_tile_size
                assert tile.access_type != L2AccessType.OUTPUT_SCALE


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
