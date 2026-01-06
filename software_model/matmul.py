from utils import size
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, inf
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim

class Matmul(Operator):
    def __init__(self, activation_data_type: DataType, weight_data_type: DataType, intermediate_data_type: DataType, device="Orin"):
        super().__init__(0, 0, 0, 0, None)
        self.activation_data_type = activation_data_type
        self.weight_data_type = weight_data_type
        self.intermediate_data_type = intermediate_data_type
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

        # performance counters
        self.systolic_array_fma_count = 0
        self.vector_fma_count = 0
        self.reg_access_count = 0
        self.l1_access_size = 0
        self.l2_access_size = 0
        self.mem_access_size = 0
        
        assert device in ["Orin", "Thor"], "Only support Orin and Thor!"
        assert (device == "Orin" and weight_data_type.name in ("fp16", "int8", "int4")) \
            or (device == "Thor" and weight_data_type.name in ("fp8", "fp4")), \
            "Only support fp16/int8/int4 for Orin and fp8/fp4 for Thor"
        self.device = device

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.activation_data_type == input1.data_type
        assert self.weight_data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.activation_data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.activation_data_type, self.weight_data_type, self.intermediate_data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K * self.activation_data_type.word_size + self.K * self.N * self.weight_data_type.word_size + self.M * self.N * self.activation_data_type.word_size
        # print(f'{self.M}, {self.N}, {self.K}')
        return output

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = max(
            self.flop_count / (pcb_module.compute_module.get_total_systolic_array_throughput_per_cycle(self.activation_data_type) * 2 * pcb_module.compute_module.clock_freq),
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
        ) # throughput in FMA = 2 flops
        return self.roofline_latency

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, activation_data_type: DataType, weight_data_type: DataType, intermediate_data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.activation_data_type = activation_data_type
            self.weight_data_type = weight_data_type
            self.intermediate_data_type = intermediate_data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, activation word_size(B): {self.activation_data_type.word_size}, weight word_size(B): {self.weight_data_type.word_size}, intermediate word_size(B): {self.intermediate_data_type.word_size}"
            )

    class Mapping:
        def __init__(
            self,
            cta_m: int,
            cta_n: int,
            cta_k: int,
            stages: int,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            dataflow: str = "os",
        ):
            self.cta_m = cta_m
            self.cta_n = cta_n
            self.cta_k = cta_k
            self.stages = stages
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.dataflow = dataflow

        def display(self, pcb_module: Device):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(f"cta_m: {self.cta_m}, cta_n: {self.cta_n}, cta_k: {self.cta_k}, stages: {self.stages}")
            print(
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, active_blocks_per_core: {self.l2_tile_M * self.l2_tile_N // (self.cta_m * self.cta_n * pcb_module.compute_module.core_count)}"
            )

    @staticmethod
    def all_factor_pairs(m: int):
        if m <= 0:
            raise ValueError("m should be positive integer")

        res = []
        for a in range(1, m + 1):
            if m % a == 0:
                b = m // a
                res.append((a, b))
        return res

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "heuristic-GPU",
    ):
        min_cycle_count = inf
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        assert (self.device == "Thor" and M >= 64) \
            or (self.device == "Orin" and M >= 32), \
            "at least 64/32 to fit Thor/Orin GEMM workflow"
        assert K >= 256, "at least 256 to accurately model loop-K"
        if compile_mode == "heuristic-GPU":
            i = 0
            cta_m_list = [64, 128, 256] if self.device == "Thor" else [32, 64, 128, 256]
            cta_k_list = [128, 256] if self.device == "Thor" else [32, 64]
            for cta_m in cta_m_list:
                for cta_n in [64, 128, 256]:
                    for cta_k in cta_k_list:
                        for stages in [2, 3]: # [2, 3, 4, 5, 6]: # 2 is always the best according to execution results
            # for cta_m in [128]:
            #     for cta_n in [128]:
            #         for cta_k in [128]:
            #             for stages in [2]: # [2, 3, 4, 5, 6]: # 2 is always the best according to execution results
                            if K <= cta_k * (stages - 1):
                                continue  # not enough K for pipelining
                            
                            is_2_sm_mode = self.device == "Thor" and cta_m in (128, 256)

                            l1_working_set_size = int(cta_m * cta_k * self.activation_data_type.word_size + \
                                cta_n * cta_k * self.weight_data_type.word_size) * stages
                            if self.device == "Thor": # intermediate data is in TMEM for Thor
                                l1_working_set_size += int(cta_m * cta_n * self.intermediate_data_type.word_size)
                                if is_2_sm_mode:
                                    l1_working_set_size //= 2
                            if l1_working_set_size > pcb_module.compute_module.core.SRAM_size:
                                continue # cannot fit in L1
                            l2_tile_K = cta_k

                            register_usage_per_block = -1
                            if self.device == "Thor":
                                register_usage_per_block = 1 # Thor does not use registers
                            else:
                                if self.activation_data_type.name == "fp16": # suppose HMMA.m16n8k16
                                    register_usage_per_block = cta_m * cta_n * self.intermediate_data_type.word_size // 4 + \
                                        (16 * 16 + 16 * 8) * self.activation_data_type.word_size // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                                elif self.activation_data_type.name == "int8": # suppose HMMA.m16n8k32
                                    register_usage_per_block = cta_m * cta_n * self.intermediate_data_type.word_size // 4 + \
                                        (16 * 32 + 32 * 8) * self.activation_data_type.word_size // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                                else:
                                    assert False, "not implemented yet"

                            max_active_blocks_per_core = min(
                                pcb_module.compute_module.core.SRAM_size // l1_working_set_size, 
                                pcb_module.compute_module.core.total_registers // register_usage_per_block
                            )

                            for active_blocks_per_core in range(max_active_blocks_per_core, 0 , -1):
                            # for active_blocks_per_core in range(2, 1 , -1):
                                l2_tile_blocks_total = active_blocks_per_core * pcb_module.compute_module.core_count
                                if is_2_sm_mode:
                                    l2_tile_blocks_total //= 2
                                for l2_tile_M_blocks, l2_tile_N_blocks in self.all_factor_pairs(l2_tile_blocks_total):
                                # for l2_tile_M_blocks, l2_tile_N_blocks in [(8, 5)]:
                                    l2_tile_M = cta_m * l2_tile_M_blocks
                                    l2_tile_N = cta_n * l2_tile_N_blocks
                                    
                                    if l2_tile_M >= M * 2 or l2_tile_N >= N * 2:
                                        continue  # unecessarily large tile

                                    l2_working_set_size = int(
                                        l2_tile_N * l2_tile_K * self.weight_data_type.word_size
                                        + l2_tile_M * l2_tile_K * self.activation_data_type.word_size
                                    ) * stages
                                    if l2_working_set_size > pcb_module.compute_module.l2_size:
                                        continue  # cannot fit in L2

                                    i += 1
                                    start = time.time()
                                    mapping = self.Mapping(
                                        cta_m,
                                        cta_n,
                                        cta_k,
                                        stages,
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                    )
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                        is_2_sm_mode
                                    )
                                    end = time.time()
                                    # if i % 1000 == 0:
                                    #     print(f"{i} simulation time: {end-start}")
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
        self.best_mapping = best_mapping
        if self.best_mapping is not None:
            self.best_mapping.display(pcb_module)
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        is_2_sm_mode: bool
    ) -> int:
        # initialize performance counters
        self.systolic_array_fma_count = 0
        self.vector_fma_count = 0
        self.reg_access_count = 0
        self.l1_access_size = 0
        self.l2_access_size = 0
        self.mem_access_size = 0

        if self.look_up_table is None:
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
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K
        stages = mapping.stages

        l2_working_set_size = int(
                            l2_tile_N * l2_tile_K * self.weight_data_type.word_size
                            + l2_tile_M * l2_tile_K * self.activation_data_type.word_size
                        ) * stages
        assert l2_working_set_size <= pcb_module.compute_module.l2_size

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        # print('-'*20)
        # print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if M_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if N_remain != 0:
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if K_remain != 0:
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if M_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if M_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if N_remain * K_remain != 0:
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                self.activation_data_type,
                self.weight_data_type,
                self.intermediate_data_type,
                mapping,
                pcb_module,
                self.look_up_table,
                is_2_sm_mode
            )

        total_cycle_count = 0
        for n in range(ceil(N / l2_tile_N)):
            for m in range(ceil(M / l2_tile_M)):

                # Prologue
                assert (K > l2_tile_K * (stages - 1))
                total_cycle_count += pcb_module.io_module.latency_cycles + pcb_module.compute_module.l2_latency_cycles
                for k in range(stages - 1):
                    total_cycle_count += l2_tiles[m, n, k].M_K_io_cycle_count + l2_tiles[m, n, k].K_N_io_cycle_count
                if self.weight_data_type.name == "int4":
                    offset_for_w4a16 = 0.03 # obtained by fitting real machine cycles
                    total_cycle_count += l2_tiles[m, n, 0].K * l2_tiles[m, n, 0].N * offset_for_w4a16 # mainly models non-overlapped dequant overhead

                flying_tile_cycles = [0] * (stages - 1)
                for k in range(ceil(K / l2_tile_K)):
                    # current tile compute latency
                    wait_ready_cycles = flying_tile_cycles[k % (stages - 1)]
                    total_cycle_count += wait_ready_cycles + l2_tiles[m, n, k].compute_cycle_count
                    # print(f"wait_ready_cycles {wait_ready_cycles}, l2_tiles[m, n, k].compute_cycle_count {l2_tiles[m, n, k].compute_cycle_count}")

                    # update flying tile
                    k_start_loading = k + stages - 1
                    if k_start_loading < ceil(K / l2_tile_K):
                        flying_tile_cycles[k_start_loading % (stages - 1)] = wait_ready_cycles +\
                                            (l2_tiles[m, n, k_start_loading].M_K_io_cycle_count + \
                                            l2_tiles[m, n, k_start_loading].K_N_io_cycle_count) * \
                                            (stages - 1) # effective bandwidth is divided by (stages - 1) flying tiles
                    flying_tile_cycles = [max(0, lat - l2_tiles[m, n, k].compute_cycle_count \
                                        - wait_ready_cycles) for lat in flying_tile_cycles]

                # Epilogue
                total_cycle_count += l2_tiles[m, n, 0].M * l2_tiles[m, n, 0].N / pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.activation_data_type, "cvt")
                offset_for_smem_reorganizing_etc = 0 if self.device == "Thor" else 0.01 # obtained by fitting real machine cycles
                total_cycle_count += l2_tiles[m, n, 0].M * l2_tiles[m, n, 0].N * offset_for_smem_reorganizing_etc # offset, mainly models data reorganizing through smem before write to DRAM. For Blackwell, smem data reorganizing is taken over asynchronously by TMA hardware.
                total_cycle_count += pcb_module.io_module.latency_cycles + pcb_module.compute_module.l2_latency_cycles + l2_tiles[m, n, 0].M_N_io_cycle_count
                self.mem_access_size += l2_tiles[m, n, 0].mem_write_size
        
        self.systolic_array_fma_count = sum(l2_tile.systolic_array_fma_count for l2_tile in l2_tiles.flat)
        if self.weight_data_type.name == "int4":
            self.vector_fma_count = sum(l2_tile.N * l2_tile.K for l2_tile in l2_tiles.flat)
        self.reg_access_count = sum(l2_tile.reg_access_count for l2_tile in l2_tiles.flat)
        self.l1_access_size = sum(l2_tile.l1_access_size for l2_tile in l2_tiles.flat)
        self.l2_access_size = sum(l2_tile.l2_access_size for l2_tile in l2_tiles.flat)
        self.mem_access_size += sum(l2_tile.mem_read_size for l2_tile in l2_tiles.flat)
        return total_cycle_count

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            activation_data_type: DataType,
            weight_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
            is_2_sm_mode: bool
        ):
            # performance counters
            self.systolic_array_fma_count = 0
            self.reg_access_count = 0
            self.l1_access_size = 0
            self.l2_access_size = 0
            self.mem_read_size = 0
            self.mem_write_size = 0

            # print(f'L2 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K

            l1_tile_M = mapping.cta_m
            l1_tile_N = mapping.cta_n
            l1_tile_K = mapping.cta_k
            stages = mapping.stages
            assert(K <= l1_tile_K)

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            self.l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            if M_l1_t * N_l1_t * K_l1_t != 0:
                self.l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if M_remain != 0:
                self.l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if N_remain != 0:
                self.l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if K_remain != 0:
                self.l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if M_remain * N_remain != 0:
                self.l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if M_remain * K_remain != 0:
                self.l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if N_remain * K_remain != 0:
                self.l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )
            if M_remain * N_remain * K_remain != 0:
                self.l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    activation_data_type,
                    weight_data_type,
                    intermediate_data_type,
                    mapping,
                    pcb_module,
                    look_up_table,
                    is_2_sm_mode
                )

            self.M_K_io_cycle_count = max(
                self.simulate_l2_tile_io_cycle_count(M, K, activation_data_type, pcb_module),
                sum(l1_tile.M_K_io_cycle_count for l1_tile in self.l1_tiles.flat)
            )
            # print(f"l2 load {self.simulate_l2_tile_io_cycle_count(M, K, activation_data_type, pcb_module)}, l1 load {sum(l1_tile.M_K_io_cycle_count for l1_tile in self.l1_tiles.flat)}")
            self.K_N_io_cycle_count = max(
                self.simulate_l2_tile_io_cycle_count(K, N, weight_data_type, pcb_module),
                sum(l1_tile.K_N_io_cycle_count for l1_tile in self.l1_tiles.flat)
            )
            # print(f"l2 load {self.simulate_l2_tile_io_cycle_count(K, N, weight_data_type, pcb_module)}, l1 load {sum(l1_tile.K_N_io_cycle_count for l1_tile in self.l1_tiles.flat)}")
            self.mem_read_size = M * K * activation_data_type.word_size + K * N * weight_data_type.word_size
            self.M_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(M, N, activation_data_type, pcb_module)
            self.mem_write_size = M * N * activation_data_type.word_size
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, mapping, pcb_module
            )

            self.systolic_array_fma_count = sum(l1_tile.systolic_array_fma_count for l1_tile in self.l1_tiles.flat)
            self.reg_access_count = sum(l1_tile.reg_access_count for l1_tile in self.l1_tiles.flat)
            self.l1_access_size = sum(l1_tile.l1_access_size for l1_tile in self.l1_tiles.flat)
            self.l2_access_size = sum(l1_tile.l2_access_size for l1_tile in self.l1_tiles.flat)

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ): # cycles to load the tile from DRAM
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
        ) -> int:

            l1_tile_M = mapping.cta_m
            l1_tile_N = mapping.cta_n

            total_cycle_count = 0
            cycles_per_core = [0] * pcb_module.compute_module.core_count
            l1_tile_idx = 0
            # ceil(K / l1_tile_K) should be 1
            for m in range(ceil(M / l1_tile_M)):
                for n in range(ceil(N / l1_tile_N)):
                    # Active blocks on the same core (SM) does not share data in L1
                    core_idx = l1_tile_idx % pcb_module.compute_module.core_count
                    cycles_per_core[core_idx] += self.l1_tiles[m, n, 0].compute_cycle_count
                    l1_tile_idx += 1
            total_cycle_count = max(cycles_per_core)

            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            activation_data_type: DataType,
            weight_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
            is_2_sm_mode: bool
        ):
            # performance counters
            self.systolic_array_fma_count = 0
            self.reg_access_count = 0
            self.l1_access_size = 0
            self.l2_access_size = 0

            # print(f'L1 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M // 2, N, K, activation_data_type, intermediate_data_type, mapping, pcb_module, look_up_table
            ) if is_2_sm_mode else self.simulate_l1_tile_compute_cycle_count(
                M, N, K, activation_data_type, intermediate_data_type, mapping, pcb_module, look_up_table
            )
            self.M_K_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M // 2, K, activation_data_type, pcb_module
            ) if is_2_sm_mode else self.simulate_l1_tile_compute_cycle_count(
                M, N, K, activation_data_type, intermediate_data_type, mapping, pcb_module, look_up_table
            )
            self.K_N_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                K, N // 2, weight_data_type, pcb_module
            ) if is_2_sm_mode else self.simulate_l1_tile_io_cycle_count(
                K, N, weight_data_type, pcb_module
            )

        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ): # cycles to load the tile from L2 to L1
            self.l2_access_size += M * N * data_type.word_size
            return ceil(
                M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            activation_data_type: DataType,
            intermediate_data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            self.systolic_array_fma_count = M * N * K
            if activation_data_type.name == "fp16": # suppose HMMA.m16n8k16
                instruction_execution_count = self.systolic_array_fma_count // (16 * 8 * 16)
                self.reg_access_count = instruction_execution_count * (16 * 16 + 8 * 16) * activation_data_type.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_data_type.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_data_type.word_size // 4
            elif activation_data_type.name == "int8": # suppose HMMA.m16n8k32
                instruction_execution_count = self.systolic_array_fma_count // (16 * 8 * 32)
                self.reg_access_count = instruction_execution_count * (16 * 32 + 8 * 32) * activation_data_type.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_data_type.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_data_type.word_size // 4
            elif activation_data_type.name == "fp8" or activation_data_type.name == "fp4":
                pass
            else:
                assert False, "not implemented yet"

            compute_cycle_count = inf
            l1_access_size = inf
            for (
                M_tiling_factor,
                N_tiling_factor, # does not model sliced-K
            ) in Matmul.all_factor_pairs(pcb_module.compute_module.core.sublane_count):
                compute_cycle_count_temp = ceil(Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    K,
                    pcb_module.compute_module.core.systolic_array.array_height,
                    pcb_module.compute_module.core.systolic_array.array_width,
                    mapping.dataflow,
                ))
                l1_access_size_temp = (M_tiling_factor * K * N + N_tiling_factor * M * K) * activation_data_type.word_size
                if compute_cycle_count_temp < compute_cycle_count:
                    compute_cycle_count = compute_cycle_count_temp
                    l1_access_size = l1_access_size_temp
                elif compute_cycle_count_temp == compute_cycle_count:
                    l1_access_size = min(l1_access_size, l1_access_size_temp)
            self.l1_access_size = l1_access_size

            return compute_cycle_count // (4 // activation_data_type.word_size)

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
                with open(
                    f"./systolic_array_model/look_up_table_{array_width}_{array_height}.csv",
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
        # print(f'end: {M} {N} {K} {array_height} {array_width} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return cycle_count

    def run_on_gpu(
        self,
    ):
        # import subprocess
        # subprocess.run(['nvidia-smi', '-q', '–d', 'CLOCK'])
        input1 = torch.randn(
            self.computational_graph.M,
            self.computational_graph.K,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input2 = torch.randn(
            self.computational_graph.K,
            self.computational_graph.N,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        latencies = []
        input1_dummy = torch.ones(4096, 4096).cuda()
        input2_dummy = torch.ones(4096, 4096).cuda()
        # warmup
        for _ in range(3):
            torch.matmul(input1_dummy, input2_dummy)
            torch.cuda.synchronize()
            time.sleep(1)
        for _ in range(self.iterations):
            # x = torch.matmul(input1_dummy, input2_dummy)  # flush the cache
            # torch.cuda.synchronize()
            start = time.time()
            output = torch.matmul(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            assert list(output.shape) == [
                self.computational_graph.M,
                self.computational_graph.N,
            ]
            latencies.append(end - start)
            # time.sleep(1)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # min(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print("GPU kernel launch overhead: ", avg_overhead * 1e3, "ms")
        print(latencies)
        return avg_overhead
