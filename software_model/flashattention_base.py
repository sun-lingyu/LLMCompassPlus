from math import ceil, inf

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import (
    DataType,
    DeviceType,
    L2AccessType,
    L2Cache,
    Tensor,
    dtype_dict,
)


class L2CacheFlashAttention(L2Cache):
    def __init__(
        self,
        l2_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        seq_len_q: int,
        seq_len_kv: int,
        head_dim: int,
        qkv_dtype: DataType,
        output_dtype: DataType,
        is_prefill: bool,
        scale_dtype: DataType = dtype_dict["fp8"],  # NVFP4 use fp8 (ue4m3) scale
        L2Cache_previous: L2Cache = None,
    ):
        super().__init__(l2_size)
        assert seq_len_q > 0 and seq_len_kv > 0 and head_dim > 0
        scale_block_size = output_dtype.scale_block_size

        # Calculate number of tiles required for each dimension
        self.seq_q_tiles = ceil(seq_len_q / L2Cache.TILE_LENGTH)
        self.seq_kv_tiles = ceil(seq_len_kv / L2Cache.TILE_LENGTH)
        self.head_q_tiles = ceil(num_heads_q * head_dim / L2Cache.TILE_LENGTH)
        self.head_kv_tiles = ceil(num_heads_kv * head_dim / L2Cache.TILE_LENGTH)
        self.head_q_scale_tiles = ceil(
            num_heads_q * head_dim / L2Cache.TILE_LENGTH / scale_block_size
            if scale_block_size
            else None
        )

        # Size of a single tile in bytes
        self.qkv_tile_size = qkv_dtype.word_size * L2Cache.TILE_LENGTH**2
        self.output_tile_size = output_dtype.word_size * L2Cache.TILE_LENGTH**2
        self.scale_tile_size = scale_dtype.word_size * self.TILE_LENGTH**2

        # Convert resident output tiles to qkv tiles
        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.output_tile_size
            q_range = num_heads_q * head_dim
            kv_range = num_heads_kv * head_dim
            while L2Cache_previous.resident_tiles:
                tile = L2Cache_previous.resident_tiles.popitem(last=False)[0]
                if tile.access_type == L2AccessType.OUTPUT:  # qkv_proj output
                    if tile.location_tuple[1] < q_range:
                        self.resident_tiles[
                            L2Cache.Tile(L2AccessType.Q, tile.coord_tuple)
                        ] = None
                    elif is_prefill:
                        # only consider KV residence for prefill
                        # if decode, KV cache will be in DRAM
                        if tile.location_tuple[1] < q_range + kv_range:
                            tile.coord_tuple[1] -= q_range
                            self.resident_tiles[
                                L2Cache.Tile(L2AccessType.K, tile.coord_tuple)
                            ] = None
                        else:
                            tile.coord_tuple[1] -= q_range + kv_range
                            self.resident_tiles[
                                L2Cache.Tile(L2AccessType.V, tile.coord_tuple)
                            ] = None
                    self.occupied_size += L2Cache_previous.output_tile_size

    def access(
        self,
        access_type: L2AccessType,
        coord_tuple: tuple[int, int],
        scope_tuple: tuple[int, int],
    ):
        # Assume Q K V all in (B=1)(S,HD) format
        height = (
            self.seq_q_tiles
            if access_type
            in (L2AccessType.Q, L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE)
            else self.seq_kv_tiles
            if access_type in (L2AccessType.K, L2AccessType.V)
            else None
        )
        width = (
            self.head_q_tiles
            if access_type in (L2AccessType.Q, L2AccessType.OUTPUT)
            else self.head_kv_tiles
            if access_type in (L2AccessType.K, L2AccessType.V)
            else self.head_q_scale_tiles
            if access_type == L2AccessType.OUTPUT_SCALE
            else None
        )
        tile_size = (
            self.qkv_tile_size
            if access_type in (L2AccessType.Q, L2AccessType.K, L2AccessType.V)
            else self.output_tile_size
            if access_type == L2AccessType.OUTPUT
            else self.scale_tile_size
        )
        assert height and width
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
                    # assert access_type not in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE)
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
            self.qkv_tile_size
            if oldest_tile.access_type
            in (L2AccessType.Q, L2AccessType.K, L2AccessType.V)
            else self.output_tile_size
            if oldest_tile.access_type == L2AccessType.OUTPUT
            else self.scale_tile_size
        )
        if oldest_tile.access_type in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE):
            mem_access_size += tile_size
        self.occupied_size -= tile_size
        self.total_mem_access_size += mem_access_size
        return mem_access_size


class FlashAttention(Operator):
    class Mapping:
        def __init__(
            self,
            cta_seq_len_q: int,
            cta_seq_len_kv: int,
            swizzle_size: int,
            gqa_packing_size: int,
        ):
            self.cta_seq_len_q = cta_seq_len_q
            self.cta_seq_len_kv = cta_seq_len_kv
            self.swizzle_size = swizzle_size
            self.gqa_packing_size = gqa_packing_size

        def display(self):
            print(f"{'-' * 10} Mapping {'-' * 10}")
            print(
                f"cta_seq_len_q: {self.cta_seq_len_q}, cta_seq_len_kv: {self.cta_seq_len_kv}, swizzle_size: {self.swizzle_size}, gqa_packing_size: {self.gqa_packing_size}"
            )

    def __init__(
        self,
        qkv_dtype: DataType,
        intermediate_dtype: DataType,
        output_dtype: DataType,
        is_causal: bool = True,
        num_splits: int = 0,
        device="Orin",
    ) -> None:
        super().__init__()
        self.qkv_dtype = qkv_dtype
        self.intermediate_dtype = intermediate_dtype
        self.output_dtype = output_dtype
        self.is_causal = is_causal
        self.num_splits = num_splits

        assert device in ["Orin", "Thor"], "Only support Orin and Thor!"
        self.device_type = DeviceType.ORIN if device == "Orin" else DeviceType.THOR
        assert (
            self.device_type == DeviceType.ORIN and self.qkv_dtype.name == "fp16"
        ) or (self.device_type == DeviceType.THOR and self.qkv_dtype.name == "fp8"), (
            "Only support fp16 for Orin and fp8 for Thor"
        )

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        assert self.qkv_dtype == q.dtype == k.dtype == v.dtype
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3, (
            "q, k, v must in (B=1)SHD format."
        )
        assert k.shape[0] == v.shape[0], "k, v seq_len must match."
        assert q.shape[1] >= k.shape[1] == v.shape[1], (
            "q heads must be more than or equal to k, v heads."
        )
        assert q.shape[1] % k.shape[1] == 0, "q heads must be multiple of k, v heads."
        assert q.shape[2] == k.shape[2] == v.shape[2], "q, k, v head_dim must match."
        if self.is_causal:
            assert q.shape[0] == k.shape[0] == v.shape[0], (
                "For causal attention, q, k, v seq_len must match."
            )
        for i in range(3):
            assert q.shape[i] > 0, "q shape dimensions must be positive."
            assert k.shape[i] == v.shape[i] > 0, (
                "k, v shape must match and be positive."
            )

        self.seq_len_q = q.shape[0]
        self.seq_len_kv = k.shape[0]
        self.num_heads_q = q.shape[1]
        self.num_heads_kv = k.shape[1]
        self.head_dim = q.shape[2]
        output_shape = (
            self.seq_len_q,
            self.num_heads_q,
            self.head_dim,
        )

        self.fma_count = 2 * (
            self.num_heads_q * self.seq_len_q * self.head_dim * self.seq_len_kv
        )  # QK^T and PV (P = softmax(QK^T))
        if self.is_causal:
            self.fma_count /= 2

        self.io_size = self.head_dim * (
            self.num_heads_q
            * self.seq_len_q
            * (self.qkv_dtype.word_size + self.output_dtype.word_size)  # Q & Output
            + self.num_heads_kv
            * self.seq_len_kv
            * self.qkv_dtype.word_size
            * 2  # K & V
        )
        if self.num_splits > 1:
            self.io_count += (
                self.num_heads_q
                * self.seq_len_q
                * self.head_dim
                * self.num_splits
                * self.qkv_dtype.word_size
                * 2
            )  # read and store partials

        self.mem_access_size = -1

        return Tensor(output_shape, self.output_dtype)

    def compile_and_simulate(
        self,
        pcb_module: Device,
        is_prefill: bool = True,
    ):
        min_cycle_count = inf
        seq_len_q = self.seq_len_q
        seq_len_kv = self.seq_len_kv
        num_heads_q = self.num_heads_q
        num_heads_kv = self.num_heads_kv
        head_dim = self.head_dim

        cta_seq_len_q_list = [32, 64, 128, 256]
        cta_seq_len_kv_list = [32, 64, 128, 256]

        gqa_group_size = num_heads_q // num_heads_kv
        assert num_heads_q % num_heads_kv == 0
        gqa_packing_size_list = sorted(
            set(
                j
                for i in range(1, int(gqa_group_size**0.5) + 1)
                if gqa_group_size % i == 0
                for j in (i, gqa_group_size // i)
            )
        )  # get all factors of gqa_group_size

        for cta_seq_len_q in cta_seq_len_q_list:
            for cta_seq_len_kv in cta_seq_len_kv_list:
                for gqa_packing_size in gqa_packing_size_list:
                    for swizzle_size in [1, 2, 4]:
                        if swizzle_size > num_heads_q // gqa_packing_size:
                            continue
                        q_tile_size = int(
                            gqa_packing_size
                            * cta_seq_len_q
                            * head_dim
                            * self.qkv_dtype.word_size
                        )
                        kv_tile_size = int(
                            cta_seq_len_kv * head_dim * self.qkv_dtype.word_size
                        )
                        intermediate_tile_size = int(
                            gqa_packing_size
                            * cta_seq_len_q
                            * cta_seq_len_kv
                            * self.intermediate_dtype.word_size
                        )
                        stages = (
                            pcb_module.compute_module.core.SRAM_size - q_tile_size
                        ) // kv_tile_size
                        if stages != 2:
                            continue

                        # Check TMEM usage
                        if self.device_type == DeviceType.THOR:
                            pass  # TODO: implement TMEM usage check for Thor

                        # Check Register usage
                        if self.device_type == DeviceType.ORIN:
                            register_usage = (
                                (
                                    (16 * 16 + 16 * 8)
                                    * self.qkv_dtype.word_size
                                    * 2  # Input operands, double buffering, suppose HMMA.m16n8k16
                                    + (
                                        gqa_packing_size
                                        * cta_seq_len_q
                                        * cta_seq_len_kv
                                        * self.intermediate_dtype.word_size
                                    )  # S tile (S = QK^T)
                                    # In FA2 without GEMM-softmax overlap, P (P = softmax(QK^T) can reuse registers of S tile.
                                    # + (
                                    #     gqa_packing_size
                                    #     * cta_seq_len_q
                                    #     * cta_seq_len_kv
                                    #     * self.qkv_dtype.word_size
                                    # )  # P tile (P = softmax(QK^T)).
                                    # # FA3 GEMM-softmax overlap: Accumulator for next GEMM0 (S) and operand for current GEMM1 (P) live concurrently.
                                    + (
                                        gqa_packing_size
                                        * cta_seq_len_q
                                        * head_dim
                                        * self.intermediate_dtype.word_size
                                    )  # O tile (O = softmax(QK^T)V)
                                )
                                // 4
                            )

                            if (
                                register_usage
                                > pcb_module.compute_module.core.total_registers
                            ):
                                continue

                        mapping = self.Mapping(
                            cta_seq_len_q=cta_seq_len_q,
                            cta_seq_len_kv=cta_seq_len_kv,
                            swizzle_size=swizzle_size,
                            gqa_packing_size=gqa_packing_size,
                        )
                        l2_status = L2CacheFlashAttention(
                            pcb_module.memory_module.l2_size,
                            num_heads_q,
                            num_heads_kv,
                            seq_len_q,
                            seq_len_kv,
                            head_dim,
                            self.qkv_dtype,
                            self.output_dtype,
                            is_prefill=is_prefill,
                        )
                        pending_write_cycle = 0
                        cycle_count, pending_write_cycle = self.simulate(
                            mapping, pcb_module, l2_status, pending_write_cycle
                        )
                        drain_cycle = ceil(
                            l2_status.drain()
                            / (
                                pcb_module.io_module.bandwidth
                                * pcb_module.io_module.bandwidth_efficiency
                                / pcb_module.compute_module.clock_freq
                            )
                        )
                        cycle_count += pending_write_cycle + drain_cycle  # clean up
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            self.best_mapping = mapping
        self.latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.mem_access_size = l2_status.total_mem_access_size
        self.best_mapping.display()
        return self.latency

    def simulate(
        self,
        mapping: Mapping,
        pcb_module: Device,
        l2_status: L2CacheFlashAttention,
        pending_write_cycle: int,
    ):
        seq_len_q = self.seq_len_q
        seq_len_kv = self.seq_len_kv
        num_heads_q = self.num_heads_q
        num_heads_kv = self.num_heads_kv
        head_dim = self.head_dim
        cta_seq_len_q = mapping.cta_seq_len_q
        cta_seq_len_kv = mapping.cta_seq_len_kv
        gqa_packing_size = mapping.gqa_packing_size
        swizzle_size = (
            mapping.swizzle_size if mapping.swizzle_size > 1 else num_heads_q
        )  # 1 means no swizzle
        total_ctas = num_heads_q // gqa_packing_size * ceil(seq_len_q / cta_seq_len_q)
        assert num_heads_q % gqa_packing_size == 0

        cta_sequence = []
        # longest-processing-time-first scheduling of swizzled CTAs
        for head_base in range(0, num_heads_q // gqa_packing_size, swizzle_size):
            for seq_len_q_start in range(
                ceil(seq_len_q / cta_seq_len_q) - cta_seq_len_q, -1, -cta_seq_len_q
            ):  # longest-processing-time-first during causal attention
                for head_offset in range(swizzle_size):
                    head_start = (head_base + head_offset) * gqa_packing_size
                    if head_start < num_heads_q:
                        cta_sequence.append((head_start, seq_len_q_start))
        assert len(cta_sequence) == total_ctas

        # Prepare L1 Tile Simulators
        shared_l1_simulator_args = (
            head_dim,
            self.qkv_dtype,
            self.intermediate_dtype,
            self.output_dtype,
            pcb_module,
        )
        if seq_len_kv // cta_seq_len_kv > 0:
            normal_l1_tile = FlashAttention.L1TileSimulator(
                cta_seq_len_q * gqa_packing_size,
                cta_seq_len_kv,
                *shared_l1_simulator_args,
            )
        if seq_len_kv % cta_seq_len_kv > 0:
            tail_l1_tile = FlashAttention.L1TileSimulator(
                cta_seq_len_q * gqa_packing_size,
                seq_len_kv % cta_seq_len_kv,
                *shared_l1_simulator_args,
            )

        # ----------------Begin Counting cycles-------------------
        active_l1_tile_list = []
        total_cycle_count = 0
        while active_l1_tile_list:
            pass

    # There is no L2TileSimulator for FlashAttention, since the ctas does not work in lockstep during causal attention.
    # Instead we need CTASimulator to track all the CTAs.

    class CTASimulator:
        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_start: int,
            seq_len_q_start: int,
            normal_l1_tile: "FlashAttention.L1TileSimulator",
            tail_l1_tile: "FlashAttention.L1TileSimulator",
            is_causal: bool,
            l2_status: L2CacheFlashAttention,
            pcb_module: Device,
        ):
            self.seq_len_q = seq_len_q
            self.seq_len_kv = seq_len_kv
            self.head_start = head_start
            self.seq_len_q_start = seq_len_q_start
            self.normal_l1_tile = normal_l1_tile
            self.tail_l1_tile = tail_l1_tile
            self.l2_status = l2_status
            self.pcb_module = pcb_module

            if is_causal:
                assert seq_len_q == seq_len_kv # Prefill only

            l1_tile_last_idx = 

            self.l1_tiles = []

            self.curr_kv_idx = 0  # which KV tile to load next

        def processing_tail_kv(self):
            return (self.curr_kv_idx + 1) * self.cta_seq_len_kv > self.seq_len_kv

        def execute_next_kv_tile(self):
            pass

        def get_Q_io_cycle_count(self):
            l1_tile = (
                self.tail_l1_tile if self.processing_tail_kv() else self.normal_l1_tile
            )  # essentially the same for Q. keep for code consistency
            Q_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                L2AccessType.Q,
                (
                    self.seq_len_q_start,
                    self.head_start,
                ),
                (
                    l1_tile.seq_len_q,
                    l1_tile.gqa_packing_size * l1_tile.head_dim,
                ),
                self.pcb_module,
            )
            return max(Q_io_cycle_count, self.l1_Q_io_cycle_count)

        def get_KV_io_cycle_count(self, seq_len_kv_start: int):
            l1_tile = (
                self.tail_l1_tile if self.processing_tail_kv() else self.normal_l1_tile
            )
            shared_kv_io_args = (
                (
                    seq_len_kv_start,
                    self.head_start,
                ),
                (
                    l1_tile.seq_len_kv,
                    l1_tile.head_dim,
                ),
                self.pcb_module,
            )
            K_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                L2AccessType.K, shared_kv_io_args
            )
            V_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                L2AccessType.V, shared_kv_io_args
            )
            return max(
                K_io_cycle_count + V_io_cycle_count,
            )

        def simulate_l2_tile_io_cycle_count(
            self,
            access_type: L2AccessType,
            coord_tuple: tuple[int, int],
            scope_tuple: tuple[int, int],
            pcb_module: Device,
        ):  # cycles to load the tile from DRAM to l2
            mem_access_size = self.l2_status.access(
                access_type, coord_tuple, scope_tuple
            )
            mem_access_cycle = ceil(
                mem_access_size
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                )
            )
            return mem_access_cycle

    class L1TileSimulator:
        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            gqa_packing_size: int,
            head_dim: int,
            qkv_dtype: DataType,
            intermediate_dtype: DataType,
            output_dtype: DataType,
            pcb_module: Device,
        ):
            self.seq_len_q = seq_len_q
            self.seq_len_kv = seq_len_kv
            self.gqa_packing_size = gqa_packing_size
            self.head_dim = head_dim
            self.pcb_module = pcb_module

            seq_len_q_packed = seq_len_q * gqa_packing_size

            self.Q_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_q_packed, head_dim, qkv_dtype, pcb_module
            )
            self.KV_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_kv, head_dim, qkv_dtype, pcb_module
            )
            self.output_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_q_packed, head_dim, output_dtype, pcb_module
            )
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                seq_len_q_packed,
                seq_len_kv,
                head_dim,
                qkv_dtype,
                intermediate_dtype,
                pcb_module,
            )

        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, dtype: DataType, pcb_module: Device
        ):  # cycles to load the tile from L2 to L1
            return ceil(
                M
                * N
                * dtype.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            qkv_dtype: DataType,
            intermediate_dtype: DataType,
            pcb_module: Device,
        ):
            assert seq_len_q > pcb_module.compute_module.core.systolic_array.array_width
            assert head_dim > pcb_module.compute_module.core.systolic_array.array_height
            assert head_dim >= 32
            vector_unit = pcb_module.compute_module.core.vector_unit
            sublane_count = pcb_module.compute_module.core.sublane_count
            array_height = pcb_module.compute_module.core.systolic_array.array_height
            array_width = pcb_module.compute_module.core.systolic_array.array_width
            systolic_array_throughput = array_height * array_width * sublane_count
            fma_throughput = (
                vector_unit.get_throughput_per_cycle(
                    data_type=qkv_dtype, operation="fma"
                )
            ) * sublane_count
            cmp_throughput = (
                vector_unit.get_throughput_per_cycle(
                    data_type=qkv_dtype, operation="cmp"
                )
            ) * sublane_count
            add_throughput = (
                vector_unit.get_throughput_per_cycle(
                    data_type=qkv_dtype, operation="add"
                )
            ) * sublane_count

            # QK^T and PV on GEMM unit
            gemm_cycle_count = (
                seq_len_q
                * seq_len_kv
                * head_dim
                // systolic_array_throughput
                // (4 // qkv_dtype.word_size)  # systolic_array_throughput is in fp32
            )

            # Exp2 on ALU
            sfu_cycle_count = (
                seq_len_q
                * seq_len_kv
                / (
                    vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_dtype, operation="exp2"
                    )
                    * sublane_count
                )
            )

            # Other ops on ALU
            alu_cycle_count = (
                seq_len_q
                * seq_len_kv
                * (
                    1 / cmp_throughput  # Before exp2: get max of each row
                    + 1 / fma_throughput  # Before exp2: subtract max and scale by log2e
                    + 1 / add_throughput  # After exp2: sum of each row
                    + 1 / fma_throughput  # After exp2: divide by sum
                )
            )
            alu_cycle_count += (
                seq_len_q * head_dim / fma_throughput
            )  # After exp2: O rescale

            # return max(gemm_cycle_count * 2, sfu_cycle_count, alu_cycle_count) # FA3 GEMM-softmax overlap
            return gemm_cycle_count * 2 + max(
                sfu_cycle_count, alu_cycle_count
            )  # FA2 without GEMM-softmax overlap
