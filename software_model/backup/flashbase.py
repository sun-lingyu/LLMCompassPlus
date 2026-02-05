from __future__ import annotations

import os
from abc import abstractmethod
from math import ceil, floor, log2

import numpy as np
import pandas as pd
from scalesim.scale_sim import scalesim

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import (
    DataType,
    L2AccessType,
    L2Cache,
    Tensor,
    data_type_dict,
)


class L2CacheFlashAttention(L2Cache):
    """
    Simulates the L2 cache behavior specifically for FlashAttention.
    Manages tile residence and eviction policies.
    """

    def __init__(
        self,
        l2_size: int,
        batch_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        seq_len_q: int,
        seq_len_kv: int,
        head_dim: int,
        qkv_data_type: DataType,
        output_data_type: DataType,
        L2Cache_previous: L2Cache = None,
    ):
        super().__init__(l2_size)
        assert seq_len_q > 0 and seq_len_kv > 0 and head_dim > 0

        self.batch_size = batch_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        # Calculate number of tiles required for each dimension
        self.seq_len_q_tiles = ceil(seq_len_q / L2Cache.TILE_LENGTH)
        self.seq_len_kv_tiles = ceil(seq_len_kv / L2Cache.TILE_LENGTH)
        self.head_dim_tiles = ceil(head_dim / L2Cache.TILE_LENGTH)

        # Size of a single tile in bytes
        self.qkv_tile_size = qkv_data_type.word_size * L2Cache.TILE_LENGTH**2
        self.output_tile_size = output_data_type.word_size * L2Cache.TILE_LENGTH**2
        self.qkv_data_type = qkv_data_type
        self.output_data_type = output_data_type

        # Carry over resident tiles from a previous cache state if provided
        if L2Cache_previous:
            assert L2Cache_previous.output_tile_size == self.qkv_data_type
            while L2Cache_previous.resident_tiles:
                tile = L2Cache_previous.resident_tiles.popitem(last=False)
                if tile.access_type == L2AccessType.OUTPUT:
                    self.resident_tiles[
                        L2Cache.Tile(L2AccessType.OUTPUT, tile.coord_tuple)
                    ] = None
                    self.occupied_size += L2Cache_previous.output_tile_size

    def access(
        self,
        access_type: L2AccessType,
        coord_tuple: tuple[int, int, int],
        scope_tuple: tuple[int, int],
    ):
        """
        Simulates an access to the L2 cache.

        Args:
            access_type (L2AccessType): Type of data being accessed (Q, K, V, OUTPUT).
            coord_tuple (tuple): Coordinates (row, col, head_index) of the access.
            scope_tuple (tuple): Dimensions (height, width) of the access region.

        Returns:
            int: Amount of data loaded from DRAM (misses) + written back to DRAM (evictions) in bytes.
        """
        height = (
            self.seq_len_q_tiles
            if access_type == L2AccessType.Q or access_type == L2AccessType.OUTPUT
            else self.seq_len_kv_tiles
            if access_type == L2AccessType.K or access_type == L2AccessType.V
            else None
        )
        width = (
            self.head_dim_tiles
            if access_type == L2AccessType.Q or access_type == L2AccessType.OUTPUT
            else self.head_dim_tiles
            if access_type == L2AccessType.K or access_type == L2AccessType.V
            else None
        )
        num_heads = (
            self.num_heads_q
            if access_type == L2AccessType.Q or access_type == L2AccessType.OUTPUT
            else self.num_heads_kv
        )
        tile_size = (
            self.qkv_tile_size
            if access_type in [L2AccessType.Q, L2AccessType.K, L2AccessType.V]
            else self.output_tile_size
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
        assert coord_tuple[2] >= 0 and coord_tuple[2] < self.batch_size * num_heads
        mem_access_size = 0
        for i in range(
            coord_tuple[0], coord_tuple[0] + scope_tuple[0], L2Cache.TILE_LENGTH
        ):
            for j in range(
                coord_tuple[1], coord_tuple[1] + scope_tuple[1], L2Cache.TILE_LENGTH
            ):
                tile = self.Tile3D(access_type, (i, j, coord_tuple[2]))
                if tile in self.resident_tiles:  # HIT
                    # assert access_type not in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE)
                    self.resident_tiles.move_to_end(tile)  # update LRU
                else:
                    while self.occupied_size + tile_size > self.l2_size:  # EVICT
                        mem_access_size += self.evict_oldest_tile()
                    if access_type != L2AccessType.OUTPUT:  # load from DRAM
                        mem_access_size += tile_size
                    self.occupied_size += tile_size
                    self.resident_tiles[
                        self.Tile3D(access_type, (i, j, coord_tuple[2]))
                    ] = None
        self.total_mem_access_size += mem_access_size
        return mem_access_size

    def evict_oldest_tile(self):
        """
        Evicts the least recently used (LRU) tile from the cache.

        Returns:
            int: Size of the evicted tile in bytes (if it needs to be written back to DRAM).
        """
        assert self.resident_tiles

        mem_access_size = 0
        oldest_tile = self.resident_tiles.popitem(last=False)[0]
        tile_size = (
            self.qkv_tile_size
            if oldest_tile.access_type
            in (L2AccessType.Q, L2AccessType.K, L2AccessType.V)
            else self.output_tile_size
            if oldest_tile.access_type == L2AccessType.OUTPUT
            else None
        )
        if oldest_tile.access_type == L2AccessType.OUTPUT:
            mem_access_size += tile_size
        self.occupied_size -= tile_size
        self.total_mem_access_size += mem_access_size
        return mem_access_size


class FlashAttention(Operator):
    """Analytical FlashAttention simulator.

    The model follows FlashAttention's streaming algorithm: each query block is
    kept on-chip, iterates over key/value blocks, applies online softmax updates,
    and immediately accumulates the output block. The simulator captures three
    critical contributions to latency:

    * Matrix multiply compute for QK^T and P@V on systolic arrays.
    * Vector-unit work for scaling, softmax (max/exp/sum), and dropout.
    * Global I/O traffic driven by repeatedly streaming K/V blocks.

    The implementation mirrors :class:`software_model.matmul.Matmul` but keeps the
    modelling lightweight enough for analytical evaluation while still exposing
    the important knobs (blocking, causal masking, dropout).
    """

    def __init__(
        self,
        qkv_data_type: DataType,
        output_data_type: DataType,
        intermediate_data_type: DataType,
        pcb_module: Device,
        is_causal: bool = True,
        num_splits: int = 0,
    ) -> None:
        """
        Initializes the FlashAttention operator.

        Args:
            qkv_data_type (DataType): Data type for activations (Q, K, V).
            output_data_type (DataType): Data type for output.
            intermediate_data_type (DataType): Data type for intermediate calculations (e.g., accumulation).
            is_causal (bool, optional): Whether to apply causal masking. Defaults to True.
        """
        super().__init__(0, 0, 0, 0, qkv_data_type)
        self.qkv_data_type = qkv_data_type
        self.output_data_type = output_data_type
        self.intermediate_data_type = intermediate_data_type
        self.q_shape = None
        self.k_shape = None
        self.v_shape = None
        self.output_shape = None
        self.best_mapping = None
        self.look_up_table = None
        self.is_causal = is_causal
        self.num_splits = num_splits
        self.pcb_module = pcb_module

    @abstractmethod
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        raise NotImplementedError("__call__ not implemented yet.")

    @abstractmethod
    def roofline_model(self, pcb_module: Device) -> float:
        raise NotImplementedError("roofline_model not implemented yet.")

    def print_latency(self):
        print(
            f"{self.computational_graph}, {self.best_latency * 1e3:.4f} ms, {self.latency_on_gpu * 1e3:.4f} ms, {self.best_latency / self.latency_on_gpu * 100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(num_tiles_q: int, num_tiles_kv: int):
        for i in range(num_tiles_q):
            for j in range(num_tiles_kv):
                yield (i, j)

    @staticmethod
    def generate_tile_loops_with_BH(
        batchsize: int, num_heads: int, num_tiles_q: int, num_tiles_kv: int
    ):
        for b in range(batchsize):
            for h in range(num_heads):
                for i in range(num_tiles_q):
                    for j in range(num_tiles_kv):
                        yield (b, h, i, j)

    class ComputationGraph:
        """
        Represents the computational graph parameters of the FlashAttention workload.
        """

        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            batch_size: int,
            num_heads_q: int,
            num_heads_kv: int,
            num_splits: int,
            qkv_data_type: DataType,
            output_data_type: DataType,
            intermediate_data_type: DataType,
        ):
            # Sequence length of Query
            self.seq_len_q = seq_len_q
            # Sequence length of Key/Value
            self.seq_len_kv = seq_len_kv
            # Dimension of each attention head
            self.head_dim = head_dim
            # Number of Query heads
            self.num_heads_q = num_heads_q
            # Number of Key/Value heads
            self.num_heads_kv = num_heads_kv
            # Number of splits for parallel processing (FlashDecoding)
            self.num_splits = num_splits
            # Batch size
            self.batch_size = batch_size
            # Data type for Q, K, V inputs
            self.qkv_data_type = qkv_data_type
            # Data type for the output
            self.output_data_type = output_data_type
            # Data type for intermediate calculations
            self.intermediate_data_type = intermediate_data_type

        def display(self):
            print(
                f"ComputationGraph(seq_len_q={self.seq_len_q}, seq_len_kv={self.seq_len_kv}, head_dim={self.head_dim}, batch_size={self.batch_size}, num_heads_q={self.num_heads_q}, num_heads_kv={self.num_heads_kv}, num_splits={self.num_splits}, qkv_data_type={self.qkv_data_type.name}, output_data_type={self.output_data_type.name}, intermediate_data_type={self.intermediate_data_type.name})"
            )

    class Mapping:
        """
        Defines the mapping strategy for the workload on the hardware.
        Includes tiling factors for L2, L1, and register (L0) levels.
        """

        def __init__(
            self,
            l2_tile_seq_q: int,
            l2_tile_seq_kv: int,
            l1_tile_seq_q: int,
            l1_tile_seq_kv: int,
            l0_M_tiling_factor_matmul1: int,
            l0_N_tiling_factor_matmul1: int,
            l0_K_tiling_factor_matmul1: int,
            l0_M_tiling_factor_matmul2: int,
            l0_N_tiling_factor_matmul2: int,
            l0_K_tiling_factor_matmul2: int,
            dataflow: str = "os",
        ):
            # Tiling sizes for L2 cache
            self.l2_tile_seq_q = l2_tile_seq_q
            self.l2_tile_seq_kv = l2_tile_seq_kv
            # Tiling sizes for L1 cache (SRAM)
            self.l1_tile_seq_q = l1_tile_seq_q
            self.l1_tile_seq_kv = l1_tile_seq_kv
            # Tiling factors for the first matrix multiplication (QK^T)
            self.l0_M_tiling_factor_matmul1 = l0_M_tiling_factor_matmul1
            self.l0_N_tiling_factor_matmul1 = l0_N_tiling_factor_matmul1
            self.l0_K_tiling_factor_matmul1 = l0_K_tiling_factor_matmul1
            # Tiling factors for the second matrix multiplication (PV)
            self.l0_M_tiling_factor_matmul2 = l0_M_tiling_factor_matmul2
            self.l0_N_tiling_factor_matmul2 = l0_N_tiling_factor_matmul2
            self.l0_K_tiling_factor_matmul2 = l0_K_tiling_factor_matmul2

            # Dataflow strategy for the systolic array (e.g., Output Stationary 'os')
            self.dataflow = dataflow

        def display(self):
            print(
                f"Mapping(l2_tile_seq_q={self.l2_tile_seq_q}, l2_tile_seq_kv={self.l2_tile_seq_kv}, "
                f"l1_tile_seq_q={self.l1_tile_seq_q}, l1_tile_seq_kv={self.l1_tile_seq_kv},"
                f"l0_M_tiling_factor_matmul1={self.l0_M_tiling_factor_matmul1}, l0_N_tiling_factor_matmul1={self.l0_N_tiling_factor_matmul1}, l0_K_tiling_factor_matmul1={self.l0_K_tiling_factor_matmul1}, "
                f"l0_M_tiling_factor_matmul2={self.l0_M_tiling_factor_matmul2}, l0_N_tiling_factor_matmul2={self.l0_N_tiling_factor_matmul2}, l0_K_tiling_factor_matmul2={self.l0_K_tiling_factor_matmul2}, "
                f"dataflow={self.dataflow})"
            )

    def compile_and_simulate(
        self,
    ):
        """
        Compiles the workload and simulates it to find the best mapping and latency.

        This method iterates through possible tiling configurations (L2 and L1 tile sizes)
        and mapping strategies to minimize the cycle count.

        Returns:
            float: The best estimated latency in seconds.
        """
        min_cycle_count = 2**63 - 1
        best_mapping = None

        # Iterate over possible L2 tile sizes for Q (log2 scale)
        for l2_tile_seq_len_q_log2 in range(
            min(5, floor(log2(self.computational_graph.seq_len_q))),
            ceil(log2(self.computational_graph.seq_len_q)) + 1,
        ):
            l2_tile_seq_len_q = 2**l2_tile_seq_len_q_log2

            # Determine L2 tile size for KV based on splits
            l2_tile_seq_len_kv = ceil(
                self.computational_graph.seq_len_kv / self.num_splits
            )
            l2_tile_seq_len_kv_log2 = floor(log2(l2_tile_seq_len_kv))

            # Iterate over possible L1 tile sizes for Q
            for l1_tile_seq_len_q_log2 in range(
                min(5, l2_tile_seq_len_q_log2), l2_tile_seq_len_q_log2 + 1
            ):
                l1_tile_seq_len_q = 2**l1_tile_seq_len_q_log2
                if l1_tile_seq_len_q > l2_tile_seq_len_q:
                    continue

                # Iterate over possible L1 tile sizes for KV
                for l1_tile_seq_len_kv_log2 in range(
                    min(5, l2_tile_seq_len_kv_log2), l2_tile_seq_len_kv_log2 + 1
                ):
                    l1_tile_seq_len_kv = 2**l1_tile_seq_len_kv_log2
                    if l1_tile_seq_len_kv > l2_tile_seq_len_kv:
                        continue

                    # Calculate working set size to check if it fits in L2 and L1 (SRAM)
                    working_set_size_bytes = (
                        l1_tile_seq_len_q
                        * self.computational_graph.head_dim
                        * max(
                            self.pcb_module.compute_module.core_count
                            / ceil(l2_tile_seq_len_q / l1_tile_seq_len_q),
                            1,
                        )
                        * 2
                        * self.qkv_data_type.word_size
                        + l1_tile_seq_len_kv
                        * self.computational_graph.head_dim
                        * min(
                            self.pcb_module.compute_module.core_count,
                            ceil(l2_tile_seq_len_kv / l1_tile_seq_len_kv),
                        )
                        * 2
                        * self.output_data_type.word_size
                    )

                    # Constraint: Working set must fit in L2 and be larger than SRAM (otherwise it's just L1 resident?)
                    # The logic implies checking if L2 is overflowed, or if it's too small for SRAM (?) - actually checking bounds.
                    if (
                        working_set_size_bytes > self.pcb_module.compute_module.l2_size
                        or working_set_size_bytes
                        < self.pcb_module.compute_module.core.SRAM_size
                    ):
                        continue

                    # Constraint: Working set must fit in SRAM (L1)
                    if (
                        3
                        * l1_tile_seq_len_q
                        * self.computational_graph.head_dim
                        * self.qkv_data_type.word_size
                        + 4
                        * l1_tile_seq_len_kv
                        * self.computational_graph.head_dim
                        * self.output_data_type.word_size
                        + l1_tile_seq_len_q * l1_tile_seq_len_kv
                    ) > self.pcb_module.compute_module.core.SRAM_size:
                        continue

                    # Two Matmuls in FlashAttention which requires two sets of L0 tiling factors
                    # We use a fixed configuration for L0 tiling here for simplicity,
                    # but one could iterate over permutations as commented out code suggests.

                    mapping = self.Mapping(
                        l2_tile_seq_q=l2_tile_seq_len_q,
                        l2_tile_seq_kv=l2_tile_seq_len_kv,
                        l1_tile_seq_q=l1_tile_seq_len_q,
                        l1_tile_seq_kv=l1_tile_seq_len_kv,
                        l0_M_tiling_factor_matmul1=4,
                        l0_N_tiling_factor_matmul1=1,
                        l0_K_tiling_factor_matmul1=1,
                        l0_M_tiling_factor_matmul2=4,
                        l0_N_tiling_factor_matmul2=1,
                        l0_K_tiling_factor_matmul2=1,
                    )

                    # Simulate the performance with this mapping
                    cycle_count = self.simulate(
                        self.computational_graph,
                        mapping,
                        self.pcb_module,
                    )
                    # mapping.display()
                    # print(f"Cycle Count: {cycle_count}")
                    if cycle_count < min_cycle_count:
                        min_cycle_count = cycle_count
                        best_mapping = mapping

        self.best_mapping = best_mapping
        if self.best_mapping is not None:
            self.computational_graph.display()
            self.best_mapping.display()

        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / self.pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        return self.latency

    def simulate(
        self,
        computational_graph: ComputationGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        """
        Simulates the execution of the FlashAttention workload for a specific mapping.

        This method models the behavior of the hardware, considering
        L2 tiling, double buffering, and compute/IO overlap.

        Args:
            computational_graph (ComputationGraph): The computational graph of the workload.
            mapping (Mapping): The specific tiling and mapping configuration to simulate.
            pcb_module (Device): The hardware device configuration.

        Returns:
            int: The total cycle count for the simulation.
        """
        # Load systolic array performance look-up table if not already loaded
        if self.look_up_table is None:
            csv_path = f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_width}_{pcb_module.compute_module.core.systolic_array.array_height}.csv"
            if not os.path.exists(csv_path):
                open(csv_path, "a").close()
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_width}_{pcb_module.compute_module.core.systolic_array.array_height}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )

        # Unpack parameters from computational graph and mapping
        seq_len_q = computational_graph.seq_len_q
        seq_len_kv = computational_graph.seq_len_kv
        head_dim = computational_graph.head_dim
        batch_size = computational_graph.batch_size
        num_heads_q = computational_graph.num_heads_q
        num_heads_kv = computational_graph.num_heads_kv
        qkv_data_type = computational_graph.qkv_data_type
        output_data_type = computational_graph.output_data_type
        intermediate_data_type = computational_graph.intermediate_data_type

        l2_tile_seq_q = mapping.l2_tile_seq_q
        l2_tile_seq_kv = mapping.l2_tile_seq_kv

        # Pre-compute L2 tile simulators for all tiles
        # This creates objects that know how to simulate each L2 tile's computation and IO
        l2_tiles = np.empty(
            [
                batch_size,
                num_heads_q,
                ceil(seq_len_q / l2_tile_seq_q),
                ceil(seq_len_kv / l2_tile_seq_kv),
            ],
            dtype=self.L2TileSimulator,
        )

        for i in range(ceil(seq_len_q / l2_tile_seq_q)):
            for j in range(ceil(seq_len_kv / l2_tile_seq_kv)):
                temp_l2_tile_q = min(seq_len_q - i * l2_tile_seq_q, l2_tile_seq_q)
                temp_l2_tile_kv = min(seq_len_kv - j * l2_tile_seq_kv, l2_tile_seq_kv)
                l2_tiles[:, :, i, j] = self.L2TileSimulator(
                    temp_l2_tile_q,
                    temp_l2_tile_kv,
                    i * l2_tile_seq_q,
                    j * l2_tile_seq_kv,
                    seq_len_kv - seq_len_q,
                    head_dim,
                    qkv_data_type,
                    output_data_type,
                    intermediate_data_type,
                    self.is_causal,
                    mapping,
                    pcb_module,
                    self.look_up_table,
                )

        active_l2_tile_list = []

        total_cycle_count = 0
        previous_compute_cycle_count = 0
        previous_write_output_bytes = 0

        # Initialize L2 Cache model
        l2cache = L2CacheFlashAttention(
            pcb_module.compute_module.l2_size,
            batch_size,
            num_heads_q,
            num_heads_kv,
            seq_len_q,
            seq_len_kv,
            head_dim,
            qkv_data_type,
            output_data_type,
        )

        # Schedule L2 tiles (using iterator to simulate order of execution)
        for b, h, l2_tile_i, l2_tile_j in self.generate_tile_loops_with_BH(
            batch_size,
            num_heads_q,
            ceil(seq_len_q / l2_tile_seq_q),
            ceil(seq_len_kv / l2_tile_seq_kv),
        ):
            # Check causal mask: skip tiles that are masked out
            if (l2_tile_i + 1) * l2_tile_seq_q + (
                seq_len_kv - seq_len_q
            ) >= l2_tile_j * l2_tile_seq_kv or not self.is_causal:
                active_l2_tile_list.append(
                    (b, h, l2_tile_i, l2_tile_j, l2_tiles[b, h, l2_tile_i, l2_tile_j])
                )

            # If we haven't filled all cores or reached the end, continue collecting tiles
            if (
                b == batch_size - 1
                and h == num_heads_q - 1
                and l2_tile_i == ceil(seq_len_q / l2_tile_seq_q) - 1
                and l2_tile_j == ceil(seq_len_kv / l2_tile_seq_kv) - 1
            ):
                pass
            elif len(active_l2_tile_list) < pcb_module.compute_module.core_count:
                continue

            assert len(active_l2_tile_list) <= pcb_module.compute_module.core_count

            # Process the batch of active tiles concurrently on available cores
            current_compute_cycle_count = 0
            current_read_q_bytes = 0
            current_read_kv_bytes = 0
            current_write_output_bytes = 0

            for i in range(len(active_l2_tile_list)):
                temp_l2_batch, temp_l2_head, _, _, temp_l2_tile = active_l2_tile_list[i]

                # Compute cycles for this tile (dominated by the slowest core if they vary, which they shouldn't much)
                temp_l2_tile_compute_cycle_count = temp_l2_tile.compute_cycle_count
                current_compute_cycle_count = max(
                    current_compute_cycle_count,
                    temp_l2_tile_compute_cycle_count,
                )

                # Get memory access coordinates
                read_q_tile, read_kv_tile, write_output_tile = (
                    temp_l2_tile.simulate_l2_tile_io_tile()
                )
                read_q_head = temp_l2_batch * num_heads_q + temp_l2_head
                read_kv_head = temp_l2_batch * num_heads_kv + temp_l2_head // (
                    num_heads_q // num_heads_kv
                )
                write_output_head = temp_l2_batch * num_heads_q + temp_l2_head

                # Update L2 cache state and calculate DRAM traffic
                current_read_q_bytes += l2cache.access(
                    L2AccessType.Q,
                    (read_q_tile[0][0], read_q_tile[0][1], read_q_head),
                    (read_q_tile[1][0], read_q_tile[1][1]),
                )
                current_read_kv_bytes += l2cache.access(
                    L2AccessType.K,
                    (read_kv_tile[0][0], read_kv_tile[0][1], read_kv_head),
                    (read_kv_tile[1][0], read_kv_tile[1][1]),
                )
                current_read_kv_bytes += l2cache.access(
                    L2AccessType.V,
                    (read_kv_tile[0][0], read_kv_tile[0][1], read_kv_head),
                    (read_kv_tile[1][0], read_kv_tile[1][1]),
                )
                current_write_output_bytes += l2cache.access(
                    L2AccessType.OUTPUT,
                    (
                        write_output_tile[0][0],
                        write_output_tile[0][1],
                        write_output_head,
                    ),
                    (write_output_tile[1][0], write_output_tile[1][1]),
                )

            current_read_bytes = current_read_q_bytes + current_read_kv_bytes

            # Calculate cycles for memory operations
            current_read_cycle_count = (
                ceil(
                    current_read_bytes
                    / (
                        pcb_module.io_module.bandwidth
                        * pcb_module.io_module.bandwidth_efficiency
                        / pcb_module.compute_module.clock_freq
                    )
                )
                + pcb_module.io_module.latency_cycles
            )

            previous_write_cycle_count = ceil(
                previous_write_output_bytes
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                )
            )

            # Double buffering: overlap current read/previous write with previous compute
            total_cycle_count += max(
                current_read_cycle_count + previous_write_cycle_count,
                previous_compute_cycle_count,
            )

            # Update history for next iteration
            previous_compute_cycle_count = current_compute_cycle_count
            previous_write_output_bytes = current_write_output_bytes

            active_l2_tile_list = []

        # Add the last iteration's compute and write cycles
        total_cycle_count += previous_compute_cycle_count + ceil(
            previous_write_output_bytes
            / (
                pcb_module.io_module.bandwidth
                * pcb_module.io_module.bandwidth_efficiency
                / pcb_module.compute_module.clock_freq
            )
        )

        # Add cycles for reduction if using multiple splits (FlashDecoding)
        if self.num_splits > 1:
            reduction_counts = (
                seq_len_q
                * head_dim
                * self.num_splits
                * batch_size
                * num_heads_q
                / pcb_module.compute_module.core_count
            )

            reduction_io_read_counts = (
                seq_len_q
                * head_dim
                * self.num_splits
                * batch_size
                * num_heads_q
                * qkv_data_type.word_size
            )

            reduction_io_read_cycle_counts = ceil(
                reduction_io_read_counts
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                )
            )

            reduction_io_write_counts = (
                seq_len_q
                * head_dim
                * batch_size
                * num_heads_q
                * qkv_data_type.word_size
            )

            reduction_io_write_cycle_counts = ceil(
                reduction_io_write_counts
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                )
            )

            # Compute cycles for reduction (FMA)
            reduction_compute_cycles = reduction_counts / (
                pcb_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                    qkv_data_type, "fma"
                )
                * pcb_module.compute_module.core.sublane_count
            )

            # Addressing overhead cycles
            addressing_compute_cycles = (
                12
                * reduction_counts
                / (
                    pcb_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type_dict["int32"], "fma"
                    )
                    * pcb_module.compute_module.core.sublane_count
                )
            )

            total_cycle_count += (
                reduction_io_read_cycle_counts
                + reduction_io_write_cycle_counts
                + reduction_compute_cycles
                + addressing_compute_cycles
            )

        # Add latency for data type conversion if necessary
        if output_data_type != qkv_data_type:
            # Data Type Conversion IO
            data_type_conversion_io_counts = (
                seq_len_q
                * head_dim
                * batch_size
                * num_heads_q
                * (qkv_data_type.word_size + output_data_type.word_size)
            )

            data_type_conversion_io_cycle_counts = ceil(
                data_type_conversion_io_counts
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                )
            )

            total_cycle_count += data_type_conversion_io_cycle_counts

        return total_cycle_count

    class L2TileSimulator:
        """
        Simulates the processing of a single L2 tile.

        Handles the simulation of I/O for Q, K, V, and Output between L2 and DRAM,
        as well as the computation within the tile (which involves iterating over L1 tiles
        executed on the Systolic Array and Vector Units).
        """

        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            position_len_q: int,
            position_len_kv: int,
            position_offset: int,
            head_dim: int,
            qkv_data_type: DataType,
            output_data_type: DataType,
            intermediate_data_type: DataType,
            is_causal: bool,
            mapping: FlashAttention.Mapping,
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            self.seq_len_q = seq_len_q
            self.seq_len_kv = seq_len_kv
            self.position_len_q = position_len_q
            self.position_len_kv = position_len_kv
            self.position_offset = position_offset
            self.head_dim = head_dim
            self.qkv_data_type = qkv_data_type
            self.output_data_type = output_data_type
            self.intermediate_data_type = intermediate_data_type
            self.is_causal = is_causal
            self.mapping = mapping
            self.pcb_module = pcb_module
            self.look_up_table = look_up_table
            self.read_q_tile = None
            self.read_kv_tile = None
            self.write_output_tile = None

            # Calculate IO cycles for loading Q, KV and storing Output (L2 <-> DRAM)
            self.q_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                seq_len_q, head_dim, qkv_data_type, pcb_module
            )
            self.kv_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                seq_len_kv, head_dim, output_data_type, pcb_module
            )
            self.output_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                seq_len_q, head_dim, qkv_data_type, pcb_module
            )

            # Calculate Compute cycles (processing L1 tiles)
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                head_dim=head_dim,
                qkv_data_type=qkv_data_type,
                output_data_type=output_data_type,
                intermediate_data_type=intermediate_data_type,
                mapping=mapping,
                chiplet_module=pcb_module,
                look_up_table=look_up_table,
            )

        def simulate_l2_tile_io_tile(
            self,
        ) -> tuple[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[int, int], tuple[int, int]],
        ]:
            """
            Returns the coordinates of the tiles read/written by this L2 tile simulation.
            Used for cache simulation.
            """
            return self.read_q_tile, self.read_kv_tile, self.write_output_tile

        def simulate_l2_tile_io_cycle_count(
            self,
            seq_len: int,
            head_dim: int,
            data_type: DataType,
            chiplet_module: Device,
        ) -> int:
            """
            Calculates the number of cycles required to transfer a tile between L2 and DRAM.
            """
            return ceil(
                seq_len
                * head_dim
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    * chiplet_module.io_module.bandwidth_efficiency
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            qkv_data_type: DataType,
            output_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: FlashAttention.Mapping,
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            """
            Simulates the computation of an L2 tile by iterating over L1 tiles.

            It schedules L1 tiles and calculates the total compute cycles, considering
            L1 cache latency and overlap between L1 I/O and computation.
            """
            l1_tile_seq_q = mapping.l1_tile_seq_q
            l1_tile_seq_kv = mapping.l1_tile_seq_kv

            # Divide L2 tile into L1 tiles
            seq_len_q_l1_t = seq_len_q // l1_tile_seq_q
            seq_len_kv_l1_t = seq_len_kv // l1_tile_seq_kv
            seq_len_q_remaining = seq_len_q % l1_tile_seq_q
            seq_len_kv_remaining = seq_len_kv % l1_tile_seq_kv

            # Pre-compute L1 tile simulators
            l1_tiles = np.empty(
                [ceil(seq_len_q / l1_tile_seq_q), ceil(seq_len_kv / l1_tile_seq_kv)],
                dtype=FlashAttention.L1TileSimulator,
            )

            if l1_tile_seq_q * l1_tile_seq_kv != 0:
                l1_tiles[:seq_len_q_l1_t, :seq_len_kv_l1_t] = (
                    FlashAttention.L1TileSimulator(
                        l1_tile_seq_q,
                        l1_tile_seq_kv,
                        head_dim,
                        qkv_data_type,
                        output_data_type,
                        intermediate_data_type,
                        mapping,
                        chiplet_module,
                        look_up_table,
                    )
                )
            if seq_len_q_remaining != 0:
                l1_tiles[-1, :seq_len_kv_l1_t] = FlashAttention.L1TileSimulator(
                    seq_len_q_remaining,
                    l1_tile_seq_kv,
                    head_dim,
                    qkv_data_type,
                    output_data_type,
                    intermediate_data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if seq_len_kv_remaining != 0:
                l1_tiles[:seq_len_q_l1_t, -1] = FlashAttention.L1TileSimulator(
                    l1_tile_seq_q,
                    seq_len_kv_remaining,
                    head_dim,
                    qkv_data_type,
                    output_data_type,
                    intermediate_data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if seq_len_q_remaining * seq_len_kv_remaining != 0:
                l1_tiles[-1, -1] = FlashAttention.L1TileSimulator(
                    seq_len_q_remaining,
                    seq_len_kv_remaining,
                    head_dim,
                    qkv_data_type,
                    output_data_type,
                    intermediate_data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )

            total_cycle_count = 0

            previous_compute_cycle_count = 0

            max_l1_tile_i = -1
            total_l1_tile_q = 0
            max_l1_tile_j = -1
            total_l1_tile_kv = 0

            # Iterate over L1 tiles
            for l1_tile_i in range(ceil(seq_len_q / l1_tile_seq_q)):
                for l1_tile_j in range(ceil(seq_len_kv / l1_tile_seq_kv)):
                    current_read_cycle_count = 0
                    current_read_cycle_count += (
                        chiplet_module.compute_module.l2_latency_cycles
                    )

                    l1_tile = l1_tiles[l1_tile_i, l1_tile_j]

                    # Apply causal mask filtering
                    if (
                        self.position_len_q
                        + (l1_tile_i + 1) * l1_tile.seq_len_q
                        + self.position_offset
                        >= self.position_len_kv + l1_tile_j * l1_tile.seq_len_kv
                        or not self.is_causal
                    ):
                        pass
                    else:
                        continue

                    if l1_tile_i > max_l1_tile_i:
                        max_l1_tile_i = l1_tile_i
                        total_l1_tile_q += l1_tile.seq_len_q
                    if l1_tile_j > max_l1_tile_j:
                        max_l1_tile_j = l1_tile_j
                        total_l1_tile_kv += l1_tile.seq_len_kv

                    # Read KV from L2 to L1 (Q is usually kept stationary or streamed, here loading KV per Q)
                    current_read_cycle_count += 2 * l1_tile.kv_io_cycle_count

                    # Use double buffering: overlap read with previous compute
                    total_cycle_count += max(
                        current_read_cycle_count,
                        previous_compute_cycle_count,
                    )

                    previous_compute_cycle_count = l1_tile.compute_cycle_count
                    # Add masking overhead if on the diagonal
                    if (
                        self.position_len_q
                        + l1_tile_i * l1_tile.seq_len_q
                        + self.position_offset
                        < self.position_len_kv + (l1_tile_j + 1) * l1_tile.seq_len_kv
                        and self.is_causal
                    ):
                        previous_compute_cycle_count += l1_tile.mask_cycle_count

                # Load Q for the next row of tiles
                current_read_cycle_count += l1_tile.q_io_cycle_count

            # Cycles for quantization of output
            quant_output_cycle_count = ceil(
                seq_len_q
                * head_dim
                / (
                    chiplet_module.compute_module.core.sublane_count
                    * chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="cvt"
                    )
                )
            )

            # Cycles for writing back output to L2
            write_back_cycle_count = ceil(
                seq_len_q
                * head_dim
                * output_data_type.word_size
                / (chiplet_module.compute_module.l2_bandwidth_per_cycle)
            )

            total_cycle_count += (
                previous_compute_cycle_count
                + quant_output_cycle_count
                + write_back_cycle_count
            )

            # Store tile coordinates for cache simulation
            self.read_q_tile = ((self.position_len_q, 0), (total_l1_tile_q, head_dim))
            self.read_kv_tile = (
                (self.position_len_kv, 0),
                (total_l1_tile_kv, head_dim),
            )
            self.write_output_tile = ((self.position_len_q, 0), (seq_len_q, head_dim))

            return total_cycle_count

    class L1TileSimulator:
        """
        Simulates the processing of a single L1 tile on the Systolic Array and Vector Units.

        Handles the simulation of I/O between L2 and L1 (SRAM), and the computation logic
        for Matmuls, Softmax (Vector units), and intermediate reductions.
        """

        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            qkv_data_type: DataType,
            output_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: FlashAttention.Mapping,
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            self.seq_len_q = seq_len_q
            self.seq_len_kv = seq_len_kv
            self.head_dim = head_dim
            self.qkv_data_type = qkv_data_type
            self.output_data_type = output_data_type
            self.intermediate_data_type = intermediate_data_type
            self.mapping = mapping
            self.pcb_module = pcb_module
            self.look_up_table = look_up_table

            # Calculate IO cycles for loading Q, KV and storing output between L2 and L1
            self.q_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_q, head_dim, qkv_data_type, pcb_module
            )
            self.kv_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_kv, head_dim, qkv_data_type, pcb_module
            )
            self.output_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_q, head_dim, output_data_type, pcb_module
            )

            # Calculate Compute cycles (Systolic Array + Vector Units)
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                seq_len_q,
                seq_len_kv,
                head_dim,
                qkv_data_type,
                output_data_type,
                intermediate_data_type,
                mapping,
                pcb_module,
                look_up_table,
            )

            # Calculate cycles for applying causal mask where needed
            self.mask_cycle_count = self.simuate_l1_tile_mask_cycle_count(
                seq_len_q,
                seq_len_kv,
                intermediate_data_type,
                pcb_module,
            )

        def simulate_l1_tile_io_cycle_count(
            self,
            seq_len: int,
            head_dim: int,
            data_type: DataType,
            chiplet_module: Device,
        ) -> int:
            """
            Calculates cycles to transfer data between L2 and L1.
            """
            return ceil(
                seq_len
                * head_dim
                * data_type.word_size
                / (chiplet_module.compute_module.l2_bandwidth_per_cycle)
            )

        def simuate_l1_tile_mask_cycle_count(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            intermediate_data_type: DataType,
            chiplet_module: Device,
        ):
            """
            Calculates cycles to apply masking (e.g. causal mask) using vector units.
            """
            return ceil(
                seq_len_q
                * seq_len_kv
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="fma"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            qkv_data_type: DataType,
            output_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: FlashAttention.Mapping,
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            """
            Simulates the computation within an L1 tile.

            Includes:
            1. First Matmul (QK^T)
            2. Softmax operations (Max, Exp, Sum)
            3. Second Matmul (PV)
            4. Output update
            """
            # Ensure contents fit in SRAM (L1)
            assert (
                3 * seq_len_q * head_dim
                + 4 * seq_len_kv * head_dim
                + seq_len_q * seq_len_kv
            ) <= chiplet_module.compute_module.core.SRAM_size // qkv_data_type.word_size

            l0_M_tiling_factor_matmul1 = mapping.l0_M_tiling_factor_matmul1
            l0_N_tiling_factor_matmul1 = mapping.l0_N_tiling_factor_matmul1
            l0_K_tiling_factor_matmul1 = mapping.l0_K_tiling_factor_matmul1

            l0_M_tiling_factor_matmul2 = mapping.l0_M_tiling_factor_matmul2
            l0_N_tiling_factor_matmul2 = mapping.l0_N_tiling_factor_matmul2
            l0_K_tiling_factor_matmul2 = mapping.l0_K_tiling_factor_matmul2

            # 1. Compute QK^T
            # Systolic array computation + Reduction (if needed)
            compute_mat1_cycle_count = ceil(
                FlashAttention.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(seq_len_q / l0_M_tiling_factor_matmul1),
                    ceil(seq_len_kv / l0_N_tiling_factor_matmul1),
                    ceil(head_dim / l0_K_tiling_factor_matmul1),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    dataflow=mapping.dataflow,
                )
                // (4 // qkv_data_type.word_size)
                + (l0_K_tiling_factor_matmul1 - 1)
                * seq_len_q
                * seq_len_kv
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="reduction"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

            # 2. Softmax Logic
            # Reduction (max)
            reduce_mat1_cycle_count = (
                seq_len_q
                * seq_len_kv
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="reduction"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

            # Exp
            pointwise_mat1_cycle_count = (
                seq_len_q
                * seq_len_kv
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="exp2"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

            # Sum
            pointwise_l_cycle_count = seq_len_q / (
                chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                    data_type=intermediate_data_type, operation="fma"
                )
                * chiplet_module.compute_module.core.sublane_count
            )

            # Quantization/Conversion
            quant_p_cycle_count = (
                seq_len_q
                * seq_len_kv
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="cvt"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

            # 3. Compute (QK^T)V
            compute_mat2_cycle_count = ceil(
                FlashAttention.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(seq_len_q / l0_M_tiling_factor_matmul2),
                    ceil(head_dim / l0_N_tiling_factor_matmul2),
                    ceil(seq_len_kv / l0_K_tiling_factor_matmul2),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    dataflow=mapping.dataflow,
                )
                // (4 // qkv_data_type.word_size)
                + (l0_K_tiling_factor_matmul2 - 1)
                * seq_len_q
                * head_dim
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="reduction"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

            # 4. Output Update (FMA)
            update_output_cycle_count = (
                seq_len_q
                * head_dim
                / (
                    chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(
                        data_type=intermediate_data_type, operation="fma"
                    )
                    * chiplet_module.compute_module.core.sublane_count
                )
            )

            # Total compute cycles: Sum of stages (assuming no overlap between stages within L1 tile, but stages might be pipelined)
            # The max() suggests that some vector operations might overlap or dominated by one constraint.
            total_cycle_count = (
                compute_mat1_cycle_count
                + max(
                    quant_p_cycle_count,
                    reduce_mat1_cycle_count,
                    pointwise_mat1_cycle_count,
                )
                + pointwise_l_cycle_count
                + compute_mat2_cycle_count
                + update_output_cycle_count
            )

            return total_cycle_count

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        dataflow="os",
    ):
        """
        Estimates the cycle count for a matrix multiplication on a systolic array.

        It first checks for high utilization cases where a simple analytical model is sufficient.
        If not, it looks up the cycle count in a pre-computed table or runs a cycle-accurate simulation (ScaleSim)
        to generate the data.

        Args:
            look_up_table (pd.DataFrame): Table containing pre-computed cycle counts.
            M (int): M dimension of the matrix multiplication (MxK * KxN).
            N (int): N dimension of the matrix multiplication.
            K (int): K dimension of the matrix multiplication.
            array_height (int): Height of the systolic array.
            array_width (int): Width of the systolic array.
            dataflow (str, optional): Dataflow architecture (e.g., "os" for output stationary). Defaults to "os".

        Returns:
            int: Estimated cycle count.
        """
        # print(f'start: {M} {N} {K} {array_height} {array_width} {dataflow}')
        assert M * N * K * array_height * array_width != 0
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(M * N * K / array_height / array_width / 0.99)
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(M * N * K / array_height / array_width / 0.98)
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(M * N * K / array_height / array_width / util_rate)
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(M * N * K / array_height / array_width / util_rate)
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(M * N * K / array_height / array_width / util_rate)
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                # print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("ReadRequestBuffer: 64\n")
                    f.writelines("WriteRequestBuffer: 64\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = "./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util

                print(
                    f"Finished simulating systolic array: {M} {N} {K} {array_height} {array_width} {dataflow} => cycle_count: {cycle_count}, util_rate: {util_rate}"
                )
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count)

    @abstractmethod
    def run_on_gpu(self) -> float:
        raise NotImplementedError("GPU simulation is not implemented yet.")
