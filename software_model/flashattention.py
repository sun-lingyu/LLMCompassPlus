from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import os
import pandas as pd
import numpy as np

from utils import size
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, Tensor
from math import ceil, isfinite, log2
from scalesim.scale_sim import scalesim
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend
from flash_attn import flash_attn_func
import copy

@dataclass
class FlashAttentionBlockingConfig:
    block_m: int
    block_n: int
    onchip_bytes: int


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

    # Conservative estimate of element-wise FLOPs per attention score (excluding exp)
    _SOFTMAX_POINTWISE_FLOPS = 6
    # Reduction FLOPs per row element
    _REDUCTION_FLOPS = 2  # max and sum per row element
    # Two FP32 vectors (m_i and l_i) are tracked per query row
    _ACCUM_STATE_WORD_SIZE = 4

    def __init__(
        self,
        activation_data_type: DataType,
        weight_data_type: DataType,
        intermediate_data_type: DataType,
        is_causal: bool = True,
        is_decoding: bool = False,
        num_splits: int = 1,
    ) -> None:
        super().__init__(0, 0, 0, 0, activation_data_type)
        self.activation_data_type = activation_data_type
        self.weight_data_type = weight_data_type
        self.intermediate_data_type = intermediate_data_type
        self.q_shape = None
        self.k_shape = None
        self.v_shape = None
        self.output_shape = None
        self.best_mapping = None
        self.look_up_table = None
        self.is_causal = is_causal
        self.is_decoding = is_decoding
        self.num_splits = num_splits
        
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Configures the operator for a FlashAttention workload."""
        
        assert self.activation_data_type == q.data_type
        assert self.weight_data_type == k.data_type == v.data_type
        
        assert len(q.shape) == len(k.shape) == len(v.shape) == 4, "q, k, v must have rank 4 [batch, heads, seq, dim]."

        for i in range(len(q.shape)):
            assert q.shape[i] > 0, "q shape dimensions must be positive."
            assert k.shape[i] == v.shape[i] > 0, "k, v shape must match and be positive."

        self.batch_size = q.shape[0]
        self.num_heads_q = q.shape[1]
        self.num_heads_kv = k.shape[1]
        
        if self.is_decoding:
            assert k.shape[2] == v.shape[2], "kv sequence lengths must match in decoding mode."
        else:
            assert q.shape[2] == k.shape[2] == v.shape[2], "q, k, v sequence lengths must match in non-decoding mode."
        
        self.seq_len_q = q.shape[2]
        self.seq_len_kv = k.shape[2]
        self.head_dim = q.shape[3]
        
        self.q_shape = q.shape
        self.k_shape = k.shape
        self.v_shape = v.shape
        self.output_shape = [self.batch_size, self.num_heads_q, self.seq_len_q, self.head_dim]
        
        self.total_heads = self.batch_size * self.num_heads_q

        # FLOP bookkeeping
        self.qk_flops = (
            2 * self.total_heads * self.seq_len_q * self.seq_len_kv * self.head_dim
        )
        self.kv_flops = (
            2 * self.total_heads * self.seq_len_q * self.head_dim * self.seq_len_kv
        )
        self.exp2_flops = self.seq_len_q * self.seq_len_kv * self.total_heads
        self.reduction_flops = self.total_heads * self.seq_len_q * (self.seq_len_kv - 1)

        if self.is_causal:
            self.qk_flops /= 2
            self.kv_flops /= 2
            self.exp2_flops /= 2
            self.reduction_flops /= 2

        self.matmul_flop_count = self.qk_flops + self.kv_flops
        self.softmax_flops_count = self.exp2_flops + self.reduction_flops
        
        self.load_count = self.activation_data_type.word_size * q.size + self.weight_data_type.word_size * (k.size + v.size)
        self.store_count = self.activation_data_type.word_size * size(self.output_shape)
        self.io_count = self.load_count + self.store_count
        
        self.computational_graph = self.ComputationGraph(
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_kv,
            head_dim=self.head_dim,
            batch_size=self.batch_size,
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_heads_kv,
            activation_data_type=self.activation_data_type,
            weight_data_type=self.weight_data_type,
            intermediate_data_type=self.intermediate_data_type,
        )

        return Tensor(self.output_shape, self.activation_data_type)

    def roofline_model(self, pcb_module: Device) -> float:
        self.roofline_latency = max(
            self.matmul_flop_count / (pcb_module.compute_module.get_total_systolic_array_throughput_per_cycle(self.activation_data_type) * 2 * pcb_module.compute_module.clock_freq) + self.exp2_flops / (pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "exp2") * 2 * pcb_module.compute_module.clock_freq) + self.reduction_flops / (pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "reduction") * 2 * pcb_module.compute_module.clock_freq),
            self.io_count 
            / min(pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle * pcb_module.compute_module.clock_freq
            ),
        )
        return self.roofline_latency
    
    def print_latency(self): 
        print(
            f"{self.computational_graph}, {self.best_latency*1e3:.4f} ms, {self.latency_on_gpu*1e3:.4f} ms, {self.best_latency/self.latency_on_gpu*100:.2f}%", flush = True,
        )
    
    @staticmethod
    def generate_tile_loops(num_tiles_q: int, num_tiles_kv: int):
        for i in range(num_tiles_q):
            for j in range(num_tiles_kv):
                yield (i, j)
                
    @staticmethod
    def generate_tile_loops_with_BH(batchsize: int, num_heads: int, num_tiles_q: int, num_tiles_kv: int):
        for b in range(batchsize):
            for h in range(num_heads):
                 for i in range(num_tiles_q):
                    for j in range(num_tiles_kv):
                        yield (b, h, i, j)
    
    class ComputationGraph:
        def __init__(self, seq_len_q: int, seq_len_kv: int, head_dim: int, batch_size: int, num_heads_q: int, num_heads_kv: int, activation_data_type: DataType, weight_data_type: DataType, intermediate_data_type: DataType):
            self.seq_len_q = seq_len_q
            self.seq_len_kv = seq_len_kv
            self.head_dim = head_dim
            self.num_heads_q = num_heads_q
            self.num_heads_kv = num_heads_kv
            self.batch_size = batch_size
            self.activation_data_type = activation_data_type
            self.weight_data_type = weight_data_type
            self.intermediate_data_type = intermediate_data_type
            
        def display(self):
            print(f"ComputationGraph(seq_len_q={self.seq_len_q}, seq_len_kv={self.seq_len_kv}, head_dim={self.head_dim}, batch_size={self.batch_size}, num_heads={self.num_heads}, activation_data_type={self.activation_data_type}, weight_data_type={self.weight_data_type}, intermediate_data_type={self.intermediate_data_type})")
    class Mapping:
        def __init__(
            self,
            l2_tile_seq_q: int,
            l2_tile_seq_kv: int,
            is_l2_double_buffering: bool,
            l1_tile_seq_q: int,
            l1_tile_seq_kv: int,
            l0_M_tiling_factor_matmul1: int,
            l0_N_tiling_factor_matmul1: int,
            l0_K_tiling_factor_matmul1: int,
            l0_M_tiling_factor_matmul2: int,
            l0_N_tiling_factor_matmul2: int,
            l0_K_tiling_factor_matmul2: int,
            l2_tile_head_dim: int = -1,
            l1_tile_head_dim: int = -1,
            dataflow: str = "os",
        ):
            self.l2_tile_seq_q = l2_tile_seq_q
            self.l2_tile_seq_kv = l2_tile_seq_kv
            self.l2_tile_head_dim = l2_tile_head_dim
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_seq_q = l1_tile_seq_q
            self.l1_tile_seq_kv = l1_tile_seq_kv
            self.l1_tile_head_dim = l1_tile_head_dim
            self.l0_M_tiling_factor_matmul1 = l0_M_tiling_factor_matmul1
            self.l0_N_tiling_factor_matmul1 = l0_N_tiling_factor_matmul1
            self.l0_K_tiling_factor_matmul1 = l0_K_tiling_factor_matmul1
            self.l0_M_tiling_factor_matmul2 = l0_M_tiling_factor_matmul2
            self.l0_N_tiling_factor_matmul2 = l0_N_tiling_factor_matmul2
            self.l0_K_tiling_factor_matmul2 = l0_K_tiling_factor_matmul2
            
            self.dataflow = dataflow
            
        def display(self):
            print(
                f"Mapping(l2_tile_seq_q={self.l2_tile_seq_q}, l2_tile_seq_kv={self.l2_tile_seq_kv}, l2_tile_head_dim={self.l2_tile_head_dim}, "
                f"is_l2_double_buffering={self.is_l2_double_buffering}, l1_tile_seq_q={self.l1_tile_seq_q}, l1_tile_seq_kv={self.l1_tile_seq_kv}, "
                f"l1_tile_head_dim={self.l1_tile_head_dim}",
                f", l0_M_tiling_factor_matmul1={self.l0_M_tiling_factor_matmul1}, l0_N_tiling_factor_matmul1={self.l0_N_tiling_factor_matmul1}, l0_K_tiling_factor_matmul1={self.l0_K_tiling_factor_matmul1}, "
                f"l0_M_tiling_factor_matmul2={self.l0_M_tiling_factor_matmul2}, l0_N_tiling_factor_matmul2={self.l0_N_tiling_factor_matmul2}, l0_K_tiling_factor_matmul2={self.l0_K_tiling_factor_matmul2}, "
                f"dataflow={self.dataflow})"
            )
            
    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ):
        min_cycle_count = 2**63 - 1
        best_mapping = None
        
        # if (seq_len_q == 1 or seq_len_kv == 1) and (compile_mode == ):
        
        if compile_mode == "exhaustive":
            for l2_tile_seq_len_q_log2 in range(5, ceil(log2(self.computational_graph.seq_len_q)) + 1):
                l2_tile_seq_len_q = 2 ** l2_tile_seq_len_q_log2
                for l2_tile_seq_len_kv_log2 in range(ceil(log2(self.computational_graph.seq_len_kv)), ceil(log2(self.computational_graph.seq_len_kv)) + 1):
                    l2_tile_seq_len_kv = 2 ** l2_tile_seq_len_kv_log2
                    for l1_tile_seq_len_q_log2 in range(5, l2_tile_seq_len_q_log2 + 1):
                        l1_tile_seq_len_q = 2 ** l1_tile_seq_len_q_log2
                        if l1_tile_seq_len_q > l2_tile_seq_len_q:
                            continue
                        for l1_tile_seq_len_kv_log2 in range(5, l2_tile_seq_len_kv_log2 + 1):
                            l1_tile_seq_len_kv = 2 ** l1_tile_seq_len_kv_log2
                            if l1_tile_seq_len_kv > l2_tile_seq_len_kv:
                                continue
                            working_set_size_bytes = (
                                l1_tile_seq_len_q * self.computational_graph.head_dim * max(pcb_module.compute_module.core_count / ceil(l2_tile_seq_len_q / l1_tile_seq_len_q), 1) * 2 * self.activation_data_type.word_size
                                + l1_tile_seq_len_kv * self.computational_graph.head_dim * min(pcb_module.compute_module.core_count, ceil(l2_tile_seq_len_kv / l1_tile_seq_len_kv)) * 2 * self.weight_data_type.word_size
                            )
                            
                            if (
                                working_set_size_bytes > pcb_module.compute_module.l2_size or working_set_size_bytes < pcb_module.compute_module.core.SRAM_size
                            ):
                                continue
                            elif (working_set_size_bytes <= pcb_module.compute_module.l2_size // 2):
                                is_l2_double_buffering = True
                            else:
                                is_l2_double_buffering = False
                            
                            if (3 * l1_tile_seq_len_q * self.computational_graph.head_dim * self.activation_data_type.word_size + 4 * l1_tile_seq_len_kv * self.computational_graph.head_dim * self.weight_data_type.word_size) > pcb_module.compute_module.core.SRAM_size:
                                continue
                            
                            for (
                                l0_M_tiling_factor_matmul1,
                                l0_N_tiling_factor_matmul1,
                                l0_K_tiling_factor_matmul1,
                            ) in self.find_permutations(
                                pcb_module.compute_module.core.sublane_count
                            ):
                                for (
                                    l0_M_tiling_factor_matmul2,
                                    l0_N_tiling_factor_matmul2,
                                    l0_K_tiling_factor_matmul2,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.sublane_count
                                ):
                                    mapping = self.Mapping(
                                        l2_tile_seq_q=l2_tile_seq_len_q,
                                        l2_tile_seq_kv=l2_tile_seq_len_kv,
                                        is_l2_double_buffering=is_l2_double_buffering,
                                        l1_tile_seq_q=l1_tile_seq_len_q,
                                        l1_tile_seq_kv=l1_tile_seq_len_kv,
                                        l0_M_tiling_factor_matmul1=l0_M_tiling_factor_matmul1,
                                        l0_N_tiling_factor_matmul1=l0_N_tiling_factor_matmul1,
                                        l0_K_tiling_factor_matmul1=l0_K_tiling_factor_matmul1,
                                        l0_M_tiling_factor_matmul2=l0_M_tiling_factor_matmul2,
                                        l0_N_tiling_factor_matmul2=l0_N_tiling_factor_matmul2,
                                        l0_K_tiling_factor_matmul2=l0_K_tiling_factor_matmul2,
                                    )
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                    )
                                    mapping.display()
                                    print(f"Cycle Count: {cycle_count}")
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
        else:
            raise NotImplementedError("Only exhaustive compile mode is implemented.")

        # block_m, block_n, onchip_bytes = self._resolve_blocking(pcb_module)
        # self.blocking = FlashAttentionBlockingConfig(block_m, block_n, onchip_bytes)

        # num_query_blocks = ceil(self.seq_len_q / block_m)
        # blocks_per_head = self._blocks_per_head(num_query_blocks, block_m, block_n)
        # total_block_pairs = self.total_heads * blocks_per_head

        # qk_cycles = self._matmul_block_cycles(block_m, block_n, self.head_dim, pcb_module)
        # pv_cycles = self._matmul_block_cycles(block_m, self.head_dim, block_n, pcb_module)
        # softmax_cycles = self._softmax_block_cycles(block_m, block_n, pcb_module)

        # per_block_cycles = qk_cycles + softmax_cycles + pv_cycles
        # total_compute_cycles = per_block_cycles * total_block_pairs

        # self.total_io_bytes = self._compute_io_bytes(block_m)
        # io_cycles = self._io_cycles(self.total_io_bytes, pcb_module)

        # overhead_cycles = int(
        #     pcb_module.compute_module.overhead.matmul * pcb_module.compute_module.clock_freq
        # )
        # total_cycles = max(total_compute_cycles, io_cycles) + overhead_cycles
        
        self.best_mapping = best_mapping
        if self.best_mapping is not None:
            self.best_mapping.display()

        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency# + 0.00009
        return self.latency
    
    def simulate(
        self,
        computational_graph: ComputationGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        if self.look_up_table is None:
            csv_path = f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_width}_{pcb_module.compute_module.core.systolic_array.array_height}.csv"
            if not os.path.exists(csv_path):
                open(csv_path, 'a').close()
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
        seq_len_q = computational_graph.seq_len_q
        seq_len_kv = computational_graph.seq_len_kv
        head_dim = computational_graph.head_dim
        batch_size = computational_graph.batch_size
        num_heads_q = computational_graph.num_heads_q
        num_heads_kv = computational_graph.num_heads_kv
        activation_data_type = computational_graph.activation_data_type
        weight_data_type = computational_graph.weight_data_type
        intermediate_data_type = computational_graph.intermediate_data_type
        
        l2_tile_seq_q = mapping.l2_tile_seq_q
        l2_tile_seq_kv = mapping.l2_tile_seq_kv
        
        seq_len_q_l2_t = seq_len_q // l2_tile_seq_q
        seq_len_kv_l2_t = seq_len_kv // l2_tile_seq_kv
        seq_len_q_remaining = seq_len_q % l2_tile_seq_q
        seq_len_kv_remaining = seq_len_kv % l2_tile_seq_kv
        
        l2_tiles = np.empty(
            [batch_size, num_heads_q, ceil(seq_len_q_l2_t), ceil(seq_len_kv_l2_t)],
            dtype = self.L2TileSimulator
        )
        
        q_tile_size = np.zeros((batch_size, num_heads_q, ceil(seq_len_q / l2_tile_seq_q)), dtype=int)
        kv_tile_size = np.zeros((batch_size, num_heads_kv, ceil(seq_len_kv / l2_tile_seq_kv)), dtype=int)
        output_tile_size = np.zeros((batch_size, num_heads_q, ceil(seq_len_q / l2_tile_seq_q), ceil(seq_len_kv / l2_tile_seq_kv)), dtype=int)
        
        for i in range(ceil(seq_len_q / l2_tile_seq_q)):
            for j in range(ceil(seq_len_kv / l2_tile_seq_kv)):
                temp_l2_tile_q = min(seq_len_q - i * l2_tile_seq_q, l2_tile_seq_q)
                temp_l2_tile_kv = min(seq_len_kv - j * l2_tile_seq_kv, l2_tile_seq_kv)
                l2_tiles[:, :, i, j] = self.L2TileSimulator(
                    temp_l2_tile_q,
                    temp_l2_tile_kv,
                    i * l2_tile_seq_q,
                    j * l2_tile_seq_kv,
                    head_dim,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    self.is_causal,
                    mapping,
                    pcb_module,
                    self.look_up_table,
                )
                q_tile_size[:, :, i] = mapping.l1_tile_seq_q * head_dim
                kv_tile_size[:, :, j] = mapping.l1_tile_seq_kv * head_dim
                output_tile_size[:, :, i, j] = temp_l2_tile_q * head_dim
        
        # if l2_tile_seq_q * l2_tile_seq_kv != 0:
        #     l2_tiles[:, :, :seq_len_q_l2_t, :seq_len_kv_l2_t] = self.L2TileSimulator(
        #         l2_tile_seq_q,
        #         l2_tile_seq_kv,
        #         head_dim,
        #         data_type,
        #         mapping,
        #         pcb_module,
        #         self.look_up_table,
        #     )
        #     q_tile_size[:, :, :seq_len_q_l2_t] = l2_tile_seq_q * head_dim
        #     kv_tile_size[:, :, :seq_len_kv_l2_t] = l2_tile_seq_kv * head_dim
        #     output_tile_size[:, :, :seq_len_q_l2_t, :seq_len_kv_l2_t] = l2_tile_seq_q * head_dim
        
        # if seq_len_q_remaining != 0:
        #     l2_tiles[:, :, -1, :seq_len_kv_l2_t] = self.L2TileSimulator(
        #         seq_len_q_remaining,
        #         l2_tile_seq_kv,
        #         head_dim,
        #         data_type,
        #         mapping,
        #         pcb_module,
        #         self.look_up_table,
        #     )
        #     q_tile_size[:, :, -1] = seq_len_q_remaining * head_dim
        #     output_tile_size[:, :, -1, :seq_len_kv_l2_t] = seq_len_q_remaining * head_dim
            
        # if seq_len_kv_remaining != 0:
        #     l2_tiles[:, :, :seq_len_q_l2_t, -1] = self.L2TileSimulator(
        #         l2_tile_seq_q,
        #         seq_len_kv_remaining,
        #         head_dim,
        #         data_type,
        #         mapping,
        #         pcb_module,
        #         self.look_up_table,
        #     )
        #     kv_tile_size[:, :, -1] = seq_len_kv_remaining * head_dim
        #     output_tile_size[:, :, :seq_len_q_l2_t, -1] = l2_tile_seq_q * head_dim
            
        # if seq_len_q_remaining * seq_len_kv_remaining != 0:
        #     l2_tiles[:, :, -1, -1] = self.L2TileSimulator(
        #         seq_len_q_remaining,
        #         seq_len_kv_remaining,
        #         head_dim,
        #         data_type,
        #         mapping,
        #         pcb_module,
        #         self.look_up_table,
        #     )
        #     output_tile_size[:, :, -1, -1] = seq_len_q_remaining * head_dim
        
        active_l2_tile_list = []
        
        previous_read_l2_tiles_q = np.zeros((batch_size, num_heads_q, ceil(seq_len_q / l2_tile_seq_q)), dtype=bool)
        
        previous_read_l2_tiles_kv = np.zeros((batch_size, num_heads_kv, ceil(seq_len_kv / l2_tile_seq_kv)), dtype=bool)
        
        previous_write_l2_tiles = np.zeros((batch_size, num_heads_q, ceil(seq_len_q / l2_tile_seq_q), ceil(seq_len_kv / l2_tile_seq_kv)), dtype=bool)
        
        current_read_l2_tiles_q = np.zeros((batch_size, num_heads_q, ceil(seq_len_q / l2_tile_seq_q)), dtype=bool)
        
        current_read_l2_tiles_kv = np.zeros((batch_size, num_heads_kv, ceil(seq_len_kv / l2_tile_seq_kv)), dtype=bool)
        
        current_write_l2_tiles = np.zeros((batch_size, num_heads_q, ceil(seq_len_q / l2_tile_seq_q), ceil(seq_len_kv / l2_tile_seq_kv)), dtype=bool)
        
        total_cycle_count = 0
        previous_compute_cycle_count = 0
        
        flag = 0
        
        for (l2_tile_i, b, h, l2_tile_j) in self.generate_tile_loops_with_BH(
            ceil(seq_len_q / l2_tile_seq_q),
            batch_size,
            num_heads_q,
            ceil(seq_len_kv / l2_tile_seq_kv),
        ): 
            if (l2_tile_i + 1) * l2_tile_seq_q >= l2_tile_j * l2_tile_seq_kv or not self.is_causal:
                active_l2_tile_list.append((b, h, l2_tile_i, l2_tile_j, l2_tiles[b, h, l2_tile_i, l2_tile_j]))
            
            if b == batch_size - 1 and h == num_heads_q - 1 and l2_tile_i == ceil(seq_len_q / l2_tile_seq_q) - 1 and l2_tile_j == ceil(seq_len_kv / l2_tile_seq_kv) - 1:
                pass
            elif(
                len(active_l2_tile_list) < pcb_module.compute_module.core_count
            ):
                continue
            
            assert(len(active_l2_tile_list) <= pcb_module.compute_module.core_count)
            
            current_read_l2_tiles_q.fill(0)
            
            current_read_l2_tiles_kv.fill(0)
            
            current_write_l2_tiles.fill(0)
            
            current_compute_cycle_count = 0
            
            for i in range(len(active_l2_tile_list)):
                temp_b, temp_h, temp_l2_tile_i, temp_l2_tile_j, temp_l2_tile = active_l2_tile_list[i]
                current_read_l2_tiles_q[temp_b, temp_h, temp_l2_tile_i] = 1
                current_read_l2_tiles_kv[temp_b, (temp_h >> (self.computational_graph.num_heads_q // self.computational_graph.num_heads_kv)), temp_l2_tile_j] = 1
                current_write_l2_tiles[temp_b, temp_h, temp_l2_tile_i, temp_l2_tile_j] = 1
                temp_l2_tile_compute_cycle_count = temp_l2_tile.compute_cycle_count
                current_compute_cycle_count = max(
                    current_compute_cycle_count,
                    temp_l2_tile_compute_cycle_count,
                )
            
            current_read_count = 0
            current_read_q_count = np.sum(
                (current_read_l2_tiles_q * (~previous_read_l2_tiles_q)) * q_tile_size
            )
            
            current_read_kv_count = np.sum(
                (current_read_l2_tiles_kv * (~previous_read_l2_tiles_kv)) * kv_tile_size
            )
            
            previous_l2_write_count = np.sum(
                previous_write_l2_tiles * output_tile_size
            )
            
            current_read_count = (
                current_read_q_count
                + 2 * current_read_kv_count
            )
            
            current_read_cycle_count = ceil(
                current_read_count * activation_data_type.word_size / (pcb_module.io_module.bandwidth * pcb_module.io_module.bandwidth_efficiency / pcb_module.compute_module.clock_freq)
            )
            
            previous_write_cycle_count = ceil(
                previous_l2_write_count * activation_data_type.word_size / (pcb_module.io_module.bandwidth * pcb_module.io_module.bandwidth_efficiency / pcb_module.compute_module.clock_freq)
            )

            if mapping.is_l2_double_buffering:
                total_cycle_count += (
                    max(
                        current_read_cycle_count,
                        previous_compute_cycle_count,
                        previous_write_cycle_count,
                    )
                )
            else:
                total_cycle_count += (
                    current_read_cycle_count
                    + previous_compute_cycle_count
                    + previous_write_cycle_count
                )
            
            print(f"Current Read Count: {current_read_count}, Current Read Cycle Count: {current_read_cycle_count}, Previous Compute Cycle Count: {previous_compute_cycle_count}, Previous Write Cycle Count: {previous_write_cycle_count}, Total Cycle Count: {total_cycle_count}")
            print(f"Current_compute_cycle_count: {current_compute_cycle_count}")
            # flag = flag + 1
            
            previous_compute_cycle_count = current_compute_cycle_count
            previous_read_l2_tiles_q = copy.deepcopy(current_read_l2_tiles_q)
            previous_read_l2_tiles_kv = copy.deepcopy(current_read_l2_tiles_kv)
            previous_write_l2_tiles = copy.deepcopy(current_write_l2_tiles)
            
            active_l2_tile_list = []
            
        total_cycle_count += previous_compute_cycle_count + ceil(
            np.sum(previous_write_l2_tiles * output_tile_size)
            * activation_data_type.word_size
            / (pcb_module.io_module.bandwidth * pcb_module.io_module.bandwidth_efficiency / pcb_module.compute_module.clock_freq)
        )
        print(f"total_cycle_count: {total_cycle_count}")
        
        return total_cycle_count
        
        
    class L2TileSimulator:
        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            position_len_q: int,
            position_len_kv: int,
            head_dim: int,
            activation_data_type: DataType,
            weight_data_type: DataType,
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
            self.head_dim = head_dim
            self.activation_data_type = activation_data_type
            self.weight_data_type = weight_data_type
            self.intermediate_data_type = intermediate_data_type
            self.is_causal = is_causal
            self.mapping = mapping
            self.pcb_module = pcb_module
            self.look_up_table = look_up_table
            self.q_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                seq_len_q, head_dim, activation_data_type, pcb_module
            )
            self.kv_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                seq_len_kv, head_dim, weight_data_type, pcb_module
            )
            self.output_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                seq_len_q, head_dim, activation_data_type, pcb_module
            )
            self.compute_cycle_count = self.simuate_l2_tile_compute_cycle_count(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                head_dim=head_dim,
                activation_data_type=activation_data_type,
                weight_data_type=weight_data_type,
                intermediate_data_type=intermediate_data_type,
                mapping=mapping,
                chiplet_module=pcb_module,
                look_up_table=look_up_table,
            )
            
        def simulate_l2_tile_io_cycle_count(
            self,
            seq_len: int,
            head_dim: int,
            data_type: DataType,
            chiplet_module: Device,
        ) -> int:
            return ceil(
                seq_len * head_dim * data_type.word_size / (chiplet_module.io_module.bandwidth / chiplet_module.compute_module.clock_freq)
            )
            
        def simuate_l2_tile_compute_cycle_count(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            activation_data_type: DataType,
            weight_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: FlashAttention.Mapping,
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            l1_tile_seq_q = mapping.l1_tile_seq_q
            l1_tile_seq_kv = mapping.l1_tile_seq_kv
            
            seq_len_q_l1_t = seq_len_q // l1_tile_seq_q
            seq_len_kv_l1_t = seq_len_kv // l1_tile_seq_kv
            seq_len_q_remaining = seq_len_q % l1_tile_seq_q
            seq_len_kv_remaining = seq_len_kv % l1_tile_seq_kv
            
            l1_tiles = np.empty(
                [ceil(seq_len_q / l1_tile_seq_q), ceil(seq_len_kv / l1_tile_seq_kv)],
                dtype = FlashAttention.L1TileSimulator
            )
            
            if l1_tile_seq_q * l1_tile_seq_kv != 0:
                l1_tiles[:seq_len_q_l1_t, :seq_len_kv_l1_t] = FlashAttention.L1TileSimulator(
                    l1_tile_seq_q,
                    l1_tile_seq_kv,
                    head_dim,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if seq_len_q_remaining != 0:
                l1_tiles[-1, :seq_len_kv_l1_t] = FlashAttention.L1TileSimulator(
                    seq_len_q_remaining,
                    l1_tile_seq_kv,
                    head_dim,
                    activation_data_type,
                    weight_data_type,
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
                    activation_data_type,
                    weight_data_type,
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
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            
            total_cycle_count = 0
            
            previous_compute_cycle_count = 0
            
            previous_l1_tile_i = -1
            
            for l1_tile_i, l1_tile_j in FlashAttention.generate_tile_loops(
                ceil(seq_len_q / l1_tile_seq_q),
                ceil(seq_len_kv / l1_tile_seq_kv),
            ):
                current_read_cycle_count = 0
                l1_tile = l1_tiles[l1_tile_i, l1_tile_j]
                
                if self.position_len_q + (l1_tile_i + 1) * l1_tile.seq_len_q >= self.position_len_kv + l1_tile_j * l1_tile.seq_len_kv or not self.is_causal:
                    pass
                else:
                    continue
                
                if (previous_l1_tile_i != l1_tile_i): 
                    current_read_cycle_count += l1_tile.q_io_cycle_count
                
                current_read_cycle_count += 2 * l1_tile.kv_io_cycle_count
                
                total_cycle_count += max(
                    current_read_cycle_count,
                    previous_compute_cycle_count,
                )
                
                previous_l1_tile_i = l1_tile_i
                
                total_cycle_count += l1_tile.output_io_cycle_count # accumulate output write back time
                
                previous_compute_cycle_count = l1_tile.compute_cycle_count
                
            total_cycle_count += previous_compute_cycle_count
            
            # if seq_len_kv_l1_t > 1:
                
            
            return total_cycle_count
            
    class L1TileSimulator:
        def __init__(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            activation_data_type: DataType,
            weight_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: FlashAttention.Mapping,
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            self.seq_len_q = seq_len_q
            self.seq_len_kv = seq_len_kv
            self.head_dim = head_dim
            self.activation_data_type = activation_data_type
            self.weight_data_type = weight_data_type
            self.intermediate_data_type = intermediate_data_type
            self.mapping = mapping
            self.pcb_module = pcb_module
            self.look_up_table = look_up_table
            self.q_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_q, head_dim, activation_data_type, pcb_module
            )
            self.kv_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_kv, head_dim, weight_data_type, pcb_module
            )
            self.output_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                seq_len_q, head_dim, activation_data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                seq_len_q,
                seq_len_kv,
                head_dim,
                activation_data_type,
                weight_data_type,
                intermediate_data_type,
                mapping,
                pcb_module,
                look_up_table,
            )
            
        def simulate_l1_tile_io_cycle_count(
            self,
            seq_len: int,
            head_dim: int,
            data_type: DataType,
            chiplet_module: Device,
        ) -> int:
            return ceil(
                seq_len * head_dim * data_type.word_size / (chiplet_module.compute_module.l2_bandwidth_per_cycle)
            )
            
        def simulate_l1_tile_compute_cycle_count(
            self,
            seq_len_q: int,
            seq_len_kv: int,
            head_dim: int,
            activation_data_type: DataType,
            weight_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: FlashAttention.Mapping,
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            assert((3 * seq_len_q * head_dim + 4 * seq_len_kv * head_dim) <= chiplet_module.compute_module.core.SRAM_size // activation_data_type.word_size)
            
            l0_M_tiling_factor_matmul1 = mapping.l0_M_tiling_factor_matmul1
            l0_N_tiling_factor_matmul1 = mapping.l0_N_tiling_factor_matmul1
            l0_K_tiling_factor_matmul1 = mapping.l0_K_tiling_factor_matmul1
            
            l0_M_tiling_factor_matmul2 = mapping.l0_M_tiling_factor_matmul2
            l0_N_tiling_factor_matmul2 = mapping.l0_N_tiling_factor_matmul2
            l0_K_tiling_factor_matmul2 = mapping.l0_K_tiling_factor_matmul2
            
            # print(f"Simulating L1 Tile Compute Cycle Count: seq_len_q={seq_len_q}, seq_len_kv={seq_len_kv}, head_dim={head_dim}")
            
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
                + (l0_K_tiling_factor_matmul1 - 1)
                * seq_len_q
                * seq_len_kv
                / chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(data_type=intermediate_data_type, operation="reduction")
            )
            
            # reduction can be pipelined so it is ignored!
            reduce_mat1_cycle_count = seq_len_kv / chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(data_type=intermediate_data_type, operation="reduction")
            
            pointwise_mat1_cycle_count = seq_len_q * seq_len_kv / chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(data_type=intermediate_data_type, operation="exp2")
            
            pointwise_l_cycle_count = seq_len_kv / chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(data_type=intermediate_data_type, operation="fma")
            
            compute_mat2_cycle_count = ceil(
                FlashAttention.simulate_systolic_array_cycle_count (
                    look_up_table,
                    ceil(seq_len_q / l0_M_tiling_factor_matmul2),
                    ceil(head_dim / l0_N_tiling_factor_matmul2),
                    ceil(seq_len_kv / l0_K_tiling_factor_matmul2),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    dataflow=mapping.dataflow,
                )
                + (l0_K_tiling_factor_matmul2 - 1)
                * seq_len_q
                * head_dim
                / chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(data_type=intermediate_data_type, operation="reduction")
            )
            
            update_output_cycle_count = seq_len_q * head_dim / chiplet_module.compute_module.core.vector_unit.get_throughput_per_cycle(data_type=activation_data_type, operation="fma")
            
            total_cycle_count = (
                compute_mat1_cycle_count
                + reduce_mat1_cycle_count
                + pointwise_mat1_cycle_count
                + pointwise_l_cycle_count
                + compute_mat2_cycle_count
                + update_output_cycle_count
            )
            
            # print(compute_mat1_cycle_count, pointwise_mat1_cycle_count, compute_mat2_cycle_count)
            
            # print(f"compute_mat1_cycle_count: {compute_mat1_cycle_count}")
            # print(f"ruduce_mat1_cycle_count: {ruduce_mat1_cycle_count}")
            # print(f"pointwise_mat1_cycle_count: {pointwise_mat1_cycle_count}")
            # print(f"compute_mat2_cycle_count: {compute_mat2_cycle_count}")

            return total_cycle_count // (4 // activation_data_type.word_size)
        
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
        # print(f'start: {M} {N} {K} {array_height} {array_width} {dataflow}')
        assert M * N * K * array_height * array_width != 0
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / 0.99
                )
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / 0.98
                )
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / util_rate
                )
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / util_rate
                )
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / util_rate
                )
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

                logpath = f"./systolic_array_model/temp/"
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
                
                print(f'Finished simulating systolic array: {M} {N} {K} {array_height} {array_width} {dataflow} => cycle_count: {cycle_count}, util_rate: {util_rate}')
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

    def run_on_gpu(self) -> float:
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required to profile FlashAttention.")
        if self.q_shape is None:
            raise RuntimeError("Call the operator with tensors before running on GPU.")

        dtype = torch.float16 if self.data_type.word_size <= 2 else torch.float32
        device = torch.device("cuda")

        Q = torch.randn(*self.q_shape, dtype=dtype, device=device)
        K = torch.randn(*self.k_shape, dtype=dtype, device=device)
        V = torch.randn(*self.v_shape, dtype=dtype, device=device)
        
        q_pt = Q.transpose(1, 2)
        k_pt = K.transpose(1, 2)
        v_pt = V.transpose(1, 2)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        latencies = []

        with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(10):  # warm up
                if self.is_decoding:
                    output_ = flash_attn_func(
                        q_pt[:, :, -1:, :], k_pt, v_pt,
                        dropout_p=0.0,
                        causal=self.is_causal,
                        incremental_state=None,
                    )
                _ = flash_attn_func(
                    q_pt, k_pt, v_pt,
                    dropout_p=0.0,
                    causal=self.is_causal,
                )
                torch.cuda.synchronize()
            for _ in range(self.iterations):
                torch.cuda.synchronize()
                start_event.record()
                output = flash_attn_func(
                    q_pt, k_pt, v_pt,
                    dropout_p=0.0,
                    causal=self.is_causal,
                )
                end_event.record()
                torch.cuda.synchronize()
                output_pt = output.transpose(1, 2)
                assert list(output_pt.shape) == self.output_shape
                latencies.append(start_event.elapsed_time(end_event) / 1000)
            
        self.latency_on_gpu = min(latencies)
        
        del Q, K, V, output
        torch.cuda.empty_cache()
        return self.latency_on_gpu