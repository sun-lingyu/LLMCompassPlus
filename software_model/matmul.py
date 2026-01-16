from utils import size
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType, Device, L2AccessType, L2Cache
from math import ceil, inf
from enum import Enum
from typing import NamedTuple
from software_model.fast_grid_cover import solve_balanced_tiling
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim

class Matmul(Operator):
    class RasterOrder(Enum):
        ALONG_M = 1
        ALONG_N = 2

    class Mapping:
        def __init__(
            self,
            cta_m: int,
            cta_n: int,
            cta_k: int,
            stages: int,
            swizzle_size: int,
            raster_order: "Matmul.RasterOrder",
            is_2sm_mode: bool,
            active_ctas_per_core: int,
            dataflow: str = "os",
        ):
            self.cta_m = cta_m
            self.cta_n = cta_n
            self.cta_k = cta_k
            self.stages = stages
            self.swizzle_size = swizzle_size
            self.raster_order = raster_order
            self.is_2sm_mode = is_2sm_mode
            self.active_ctas_per_core = active_ctas_per_core
            self.dataflow = dataflow

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(f"cta_m: {self.cta_m}, cta_n: {self.cta_n}, cta_k: {self.cta_k}, stages: {self.stages}")
            print(f"swizzle_size: {self.swizzle_size}, raster_order: {"ALONG_M" if self.raster_order == Matmul.RasterOrder.ALONG_M else "ALONG_N"}")
            print(f"is_2sm_mode: {self.is_2sm_mode}, active_ctas_per_core: {self.active_ctas_per_core}")

    class L2CacheMatmul(L2Cache):
        def __init__(self,
                    l2_size: int,
                    M: int,
                    N: int,
                    K: int,
                    activation_dtype: DataType,
                    weight_dtype: DataType,
                    output_dtype: DataType,
                    L2Cache_previous: L2Cache = None
                    ):
            super().__init__(l2_size)
            # change output to activation
            assert M > 0 and N > 0 and K > 0

            self.m_tiles = ceil(M / L2Cache.TILE_LENGTH)
            self.n_tiles = ceil(N / L2Cache.TILE_LENGTH)
            self.k_tiles = ceil(K / L2Cache.TILE_LENGTH)
            self.activation_tile_size = activation_dtype.wordsize * self.TILE_LENGTH ** 2
            self.weight_tile_size = weight_dtype.wordsize * self.TILE_LENGTH ** 2
            self.output_tile_size = output_dtype.wordsize * self.TILE_LENGTH ** 2

            if L2Cache_previous:
                assert L2Cache_previous.output_tile_size == self.activation_tile_size
                while L2Cache_previous.resident_tiles:
                    tile = L2Cache_previous.resident_tiles.popitem(last=False)[0]
                    if tile.access_type == L2AccessType.OUTPUT:
                        self.resident_tiles[L2Cache.Tile(L2AccessType.ACT, tile.coord_tuple)] = None
                        self.occupied_size += L2Cache_previous.output_tile_size
        
        def access(self,
                   access_type: L2AccessType,
                   coord_tuple: tuple[int, int],
                   scope_tuple: tuple[int, int]
                   ):
            height = self.m_tiles if access_type == L2AccessType.ACTIVATION else \
                    self.k_tiles if access_type == L2AccessType.WEIGHT else \
                    self.m_tiles
            width = self.k_tiles if access_type == L2AccessType.ACTIVATION else \
                    self.n_tiles if access_type == L2AccessType.WEIGHT else \
                    self.n_tiles
            tile_size = self.activation_tile_size if access_type == L2AccessType.ACTIVATION else \
                    self.weight_tile_size if access_type == L2AccessType.WEIGHT else \
                    self.output_tile_size

            assert coord_tuple[0] % L2Cache.TILE_LENGTH == 0 and coord_tuple[1] % L2Cache.TILE_LENGTH == 0
            assert scope_tuple[0] % L2Cache.TILE_LENGTH == 0 and scope_tuple[1] % L2Cache.TILE_LENGTH == 0
            assert coord_tuple[0] >= 0 and coord_tuple[0] + scope_tuple[0] <= height * L2Cache.TILE_LENGTH
            assert coord_tuple[1] >= 0 and coord_tuple[1] + scope_tuple[0] <= width * L2Cache.TILE_LENGTH

            mem_access_size = 0
            for i in range(coord_tuple[0], coord_tuple[0] + scope_tuple[0], L2Cache.TILE_LENGTH):
                for j in range(coord_tuple[1], coord_tuple[1] + scope_tuple[1], L2Cache.TILE_LENGTH):
                    tile = self.Tile(access_type, (i, j))
                    if tile in self.resident_tiles: # HIT
                        assert access_type != L2AccessType.OUTPUT
                        self.resident_tiles.move_to_end(tile) # update LRU
                        return 0
                    else:
                        while self.occupied_size + tile_size > self.l2_size: # EVICT
                            mem_access_size += self.evict_oldest_tile()
                        if access_type != L2AccessType.OUTPUT: # load from DRAM
                            mem_access_size += tile_size
                        self.occupied_size += tile_size
                        self.resident_tiles[self.Tile(access_type, (i, j))] = None
            return mem_access_size

        def evict_oldest_tile(self):
            assert self.resident_tiles
            
            mem_access_size = 0
            oldest_tile = self.resident_tiles.popitem(last=False)[0]
            tile_size = self.activation_tile_size if oldest_tile.access_type == L2AccessType.ACTIVATION else \
                    self.weight_tile_size if oldest_tile.access_type == L2AccessType.WEIGHT else \
                    self.output_tile_size
            if oldest_tile.access_type == L2AccessType.OUTPUT:
                mem_access_size += tile_size
            self.occupied_size -= tile_size
            return mem_access_size

    def __init__(self, activation_dtype: DataType, \
                weight_dtype: DataType, \
                intermediate_dtype: DataType, \
                output_dtype: DataType, \
                device="Orin"):
        super().__init__(0, 0, 0, 0, None)
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.intermediate_dtype = intermediate_dtype
        self.output_dtype = output_dtype
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
        self.device = Device.ORIN if device =="Orin" else Device.THOR
        assert (self.device == Device.ORIN and weight_dtype.name in ("fp16", "int8", "int4")) \
            or (self.device == Device.THOR and weight_dtype.name in ("fp8", "fp4")), \
            "Only support fp16/int8/int4 for Orin and fp8/fp4 for Thor"

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.activation_dtype == input1.data_type
        assert self.weight_dtype == input2.data_type
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
        output = Tensor(self.output_shape, self.output_dtype)
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K * self.activation_dtype.word_size + self.K * self.N * self.weight_dtype.word_size + self.M * self.N * self.output_dtype.word_size
        self.fma_count = self.M * self.K * self.N
        self.mem_access_size = -1
        return output

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = max(
            self.flop_count / (pcb_module.compute_module.get_total_systolic_array_throughput_per_cycle(self.activation_dtype) * 2 * pcb_module.compute_module.clock_freq),
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            )) # throughput in FMA = 2 flops
        return self.roofline_latency

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
    ):
        min_cycle_count = inf
        M = self.M
        N = self.N
        K = self.K
        assert (self.device == Device.THOR and M >= 64) \
            or (self.device == Device.ORIN and M >= 32), \
            "at least 64/32 to fit Thor/Orin GEMM workflow"
        assert K >= 256, "at least 256 to accurately model loop-K"

        cta_m_list = [64, 128, 256] if self.device == Device.THOR else [32, 64, 128, 256]
        cta_k_list = [128] if self.device == Device.THOR else [32, 64]
        # for cta_m in cta_m_list:
        #     for cta_n in [64, 128, 256]:
        #         for cta_k in cta_k_list:
        for cta_m in [256]:
            for cta_n in [256]:
                for cta_k in [128]:
                    for swizzle_size in [1, 2, 4, 8]:
                        activation_tile_size = int(cta_m * cta_k * self.activation_dtype.word_size)
                        weight_tile_size = int(cta_n * cta_k * self.weight_dtype.word_size)
                        output_tile_size = int(cta_m * cta_n * self.output_dtype.word_size)
                        intermediate_tile_size = int(cta_m * cta_n * self.intermediate_dtype.word_size)
                        is_2sm_mode = (self.device == Device.THOR and cta_m in (128, 256)) # Blackwell 2-SM mode
                        l1_working_set_size = (activation_tile_size + weight_tile_size)
                        if is_2sm_mode:
                            l1_working_set_size //= 2

                        # Check enough stages
                        if self.device == Device.THOR:
                            # Blackwell: CUTLASS StageCountAuto Policy: 1 CTA/SM
                            stages = pcb_module.compute_module.core.SRAM_size // l1_working_set_size
                            if stages < 3:
                                continue # not enough pipeline stages
                        elif self.device == Device.ORIN:
                            # Ampere: 3 stages to maximize active_ctas_per_core
                            stages = 3

                        # Check l1 usage
                        l1_working_set_size *= stages
                        if l1_working_set_size > pcb_module.compute_module.core.SRAM_size:
                            continue

                        # Check TMEM usage
                        if self.device == Device.THOR:
                            intermediate_size = intermediate_tile_size
                            if is_2sm_mode:
                                intermediate_size //= 2
                            assert intermediate_size <= 256 * 1024 # TMEM: 256KB

                        # Check Register usage
                        register_usage = 1
                        if self.device == Device.ORIN:
                            if self.activation_dtype.name == "fp16": # suppose HMMA.m16n8k16
                                register_usage = cta_m * cta_n * self.intermediate_dtype.word_size // 4 + \
                                    (16 * 16 + 16 * 8) * self.activation_dtype.word_size // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                            elif self.activation_dtype.name == "int8": # suppose HMMA.m16n8k32
                                register_usage = cta_m * cta_n * self.intermediate_dtype.word_size // 4 + \
                                    (16 * 32 + 32 * 8) * self.activation_dtype.word_size // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                            else:
                                assert False, "not implemented yet"
                            assert register_usage <= pcb_module.compute_module.core.total_registers           

                        active_ctas_per_core = min(
                            pcb_module.compute_module.core.SRAM_size // l1_working_set_size, 
                            pcb_module.compute_module.core.total_registers // register_usage
                        )
                        assert self.device == Device.THOR and active_ctas_per_core == 1
                        
                        raster_order = Matmul.RasterOrder.ALONG_M if M <= N else Matmul.RasterOrder.ALONG_N
                        mapping = self.Mapping(
                            cta_m,
                            cta_n,
                            cta_k,
                            swizzle_size,
                            raster_order,
                            is_2sm_mode,
                            active_ctas_per_core
                        )
                        l2_status = Matmul.L2CacheMatmul(
                            pcb_module.compute_module.l2_size, 
                            ceil(M / L2Cache.TILE_LENGTH), 
                            ceil(N / L2Cache.TILE_LENGTH), 
                            ceil(K / L2Cache.TILE_LENGTH),
                            self.activation_dtype, 
                            self.weight_dtype,
                            self.output_dtype
                        )
                        cycle_count, pending_write_cycle = self.simulate(
                            mapping,
                            pcb_module,
                            l2_status,
                            pending_write_cycle
                        )
                        drain_cycle = ceil(l2_status.drain() / (
                            pcb_module.io_module.bandwidth
                            * pcb_module.io_module.bandwidth_efficiency
                            / pcb_module.compute_module.clock_freq
                        ))
                        cycle_count += pending_write_cycle + drain_cycle # clean up

                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
        self.best_cycle_count = min_cycle_count
        self.latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.best_mapping.display()
        return self.latency

    def simulate(
        self,
        mapping: Mapping,
        pcb_module: Device,
        l2_status: L2CacheMatmul,
        pending_write_cycle: int
    ) -> int:
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
        
        M = self.M
        N = self.N
        K = self.K
        execution_unit_num = pcb_module.compute_module.core_count
        if mapping.is_2sm_mode:
            execution_unit_num //= 2
        active_ctas_per_wave = mapping.active_ctas_per_core * execution_unit_num
        cta_m = mapping.cta_m
        cta_n = mapping.cta_n
        cta_k = mapping.cta_k
        raster_order = mapping.raster_order
        swizzle_size = mapping.swizzle_size
        is_2sm_mode = mapping.is_2sm_mode
        dataflow = mapping.dataflow
        total_ctas = ceil(M / cta_m) * ceil(N / cta_n)

        normal_l2_tiles = [] # l2_tile_K == cta_k
        tail_l2_tiles = [] # l2_tile_K < cta_k
        normal_l1_tile = Matmul.L1TileSimulator(
            cta_m,
            cta_n,
            cta_k,
            self.activation_dtype,
            self.weight_dtype,
            self.intermediate_dtype,
            pcb_module,
            self.look_up_table,
            is_2sm_mode,
            dataflow
        )
        tail_l1_tile = Matmul.L1TileSimulator(
            cta_m,
            cta_n,
            K % cta_k,
            self.activation_dtype,
            self.weight_dtype,
            self.intermediate_dtype,
            pcb_module,
            self.look_up_table,
            is_2sm_mode,
            dataflow
        )

        cta_sequence = []
        # L2 swizzling
        if raster_order == Matmul.RasterOrder.ALONG_M:
            for m_base in range(0, ceil(M / cta_m), swizzle_size):
                for n in range(ceil(N / cta_n)):
                    for m_offset in range(swizzle_size):
                        m = m_base + m_offset
                        if m < ceil(M / cta_m):
                            cta_sequence.append(m * cta_m, n * cta_n)
        elif raster_order == Matmul.RasterOrder.ALONG_N:
            for n_base in range(0, ceil(N / cta_n), swizzle_size):
                for m in range(ceil(M / cta_m)):
                    for n_offset in range(swizzle_size):
                        n = n_base + n_offset
                        if n < ceil(N / cta_n):
                            cta_sequence.append(m * cta_m, n * cta_n)

        for waveid, i in range(0, total_ctas, active_ctas_per_wave):
            cta_chunk = cta_sequence[i : i + self.num_sm]
            m_coords = [p[0] for p in cta_chunk]
            n_coords = [p[1] for p in cta_chunk]
            
            normal_l2_tiles.append(
                self.L2TileSimulator(

                )
            )

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K
        assert l2_tile_M % mapping.cta_m == 0 and l2_tile_N % mapping.cta_n == 0
        l2_tile_M_in_tile = l2_tile_M // mapping.cta_m
        l2_tile_N_in_tile = l2_tile_N // mapping.cta_n
        l2_coord_M_in_tile = mapping.l2_coord_M_in_tile
        l2_coord_N_in_tile = mapping.l2_coord_N_in_tile

        K_l2_t = K // l2_tile_K
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        # print('-'*20)
        # print(l2_tiles.shape)
        if K_l2_t != 0:
            l2_tiles[:K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                self.activation_dtype,
                self.weight_dtype,
                self.intermediate_dtype,
                mapping,
                pcb_module,
                self.look_up_table,
                l2_status
            )
        if K_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                self.activation_dtype,
                self.weight_dtype,
                self.intermediate_dtype,
                mapping,
                pcb_module,
                self.look_up_table,
                l2_status
            )

        total_cycle_count = 0
        # Prologue
        M_K_io_cycle_count = l2_tiles[0].simulate_l2_tile_io_cycle_count(
            L2AccessType.ACTIVATION, 
            (l2_coord_M_in_tile, 0), 
            (l2_tile_M_in_tile, 1), 
            pcb_module
        )
        K_N_io_cycle_count = l2_tiles[0].simulate_l2_tile_io_cycle_count(
            L2AccessType.WEIGHT, 
            (0, l2_coord_N_in_tile), 
            (1, l2_tile_N_in_tile), 
            pcb_module
        )
        # total_cycle_count += max(M_K_io_cycle_count, l2_tiles[0].M_K_l1_cycle_count) + \
        #     max(K_N_io_cycle_count, l2_tiles[0].K_N_l1_cycle_count)
        total_cycle_count += max(
            M_K_io_cycle_count + K_N_io_cycle_count, 
            l2_tiles[0].M_K_l1_cycle_count + l2_tiles[0].K_N_l1_cycle_count
        )
        if M_K_io_cycle_count + K_N_io_cycle_count == 0:
            pending_write_cycle = max(
                0, 
                pending_write_cycle - (l2_tiles[0].K_N_l1_cycle_count + l2_tiles[0].M_K_l1_cycle_count)
            ) # overlap pending write with compute
        else:
            total_cycle_count += pending_write_cycle
            pending_write_cycle = 0
        if self.weight_dtype.name == "int4":
            offset_for_w4a16 = 0.03 # obtained by fitting real machine cycles
            total_cycle_count += l2_tiles[0].K * l2_tiles[0].N * offset_for_w4a16 # mainly models non-overlapped dequant overhead

        # Loop K: double buffering
        wait_ready_cycle = 0
        for k in range(ceil(K / l2_tile_K)):
            # current tile compute latency
            total_cycle_count += wait_ready_cycle + l2_tiles[k].compute_cycle_count
            if M_K_io_cycle_count + K_N_io_cycle_count == 0:
                pending_write_cycle = max(
                    0, 
                    pending_write_cycle - (wait_ready_cycle + l2_tiles[k].compute_cycle_count)
                ) # overlap pending write with compute
            else:
                total_cycle_count += pending_write_cycle
                pending_write_cycle = 0
            # print(f"wait_ready_cycle {wait_ready_cycle}, l2_tiles[k].compute_cycle_count {l2_tiles[k].compute_cycle_count}")

            # update wait_ready_cycle
            if k + 1 < ceil(K / l2_tile_K):
                M_K_io_cycle_count = l2_tiles[k + 1].simulate_l2_tile_io_cycle_count(
                    L2AccessType.ACTIVATION, 
                    (l2_coord_M_in_tile, k + 1),
                    (l2_tile_M_in_tile, 1),
                    pcb_module)
                K_N_io_cycle_count = l2_tiles[k + 1].simulate_l2_tile_io_cycle_count(
                                    L2AccessType.WEIGHT, 
                                    (k + 1, l2_coord_N_in_tile),
                                    (1, l2_tile_N_in_tile),
                                    pcb_module)
                read_next_tile_cycle = max(
                    M_K_io_cycle_count + K_N_io_cycle_count, 
                    l2_tiles[k].K_N_l1_cycle_count + l2_tiles[k].M_K_l1_cycle_count
                )
                wait_ready_cycle = max(0, read_next_tile_cycle - l2_tiles[k].compute_cycle_count)
            else:
                wait_ready_cycle = 0

        # Epilogue
        total_cycle_count += l2_tiles[0].M * l2_tiles[0].N // pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.output_dtype, "cvt")
        offset_for_smem_reorganizing_etc = 0 if self.device == Device.THOR else 0.01 # obtained by fitting real machine cycles
        total_cycle_count += l2_tiles[0].M * l2_tiles[0].N * offset_for_smem_reorganizing_etc # offset, mainly models data reorganizing through smem before write to DRAM. For Blackwell, smem data reorganizing is taken over asynchronously by TMA hardware.
        M_N_io_cycle_count = l2_tiles[0].simulate_l2_tile_io_cycle_count(L2AccessType.OUTPUT, (l2_coord_M_in_tile, l2_coord_N_in_tile), (l2_tile_M_in_tile, l2_tile_N_in_tile), pcb_module)
        pending_write_cycle += max(0, M_N_io_cycle_count - l2_tiles[k].compute_cycle_count) # overlap with compute
        self.systolic_array_fma_count = sum(l2_tile.systolic_array_fma_count for l2_tile in l2_tiles.flat)
        if self.weight_dtype.name == "int4":
            self.vector_fma_count = sum(l2_tile.N * l2_tile.K for l2_tile in l2_tiles.flat)
        self.reg_access_count = sum(l2_tile.reg_access_count for l2_tile in l2_tiles.flat)
        self.l1_access_size = sum(l2_tile.l1_access_size for l2_tile in l2_tiles.flat)
        self.l2_access_size = sum(l2_tile.l2_access_size for l2_tile in l2_tiles.flat)
        self.mem_access_size += sum(l2_tile.mem_access_size for l2_tile in l2_tiles.flat)
        # print(f"total_cycle_count: {total_cycle_count}")
        return total_cycle_count, pending_write_cycle

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            coord_M: int,
            coord_N: int,
            l1_tile_count: int,
            l1_tile: "Matmul.L1TileSimulator",
            active_ctas_per_core: int,
            l2_status: "Matmul.L2CacheMatmul"
        ):
            self.l2_status = l2_status

            self.M = M
            self.N = N
            self.K = K
            self.coord_tuple = (coord_M, coord_N)
            self.size_tuple = 

            self.l1_tile_count = l1_tile_count
            self.l1_tile = l1_tile

            self.active_ctas_per_core = active_ctas_per_core

            self.M_K_l1_cycle_count = self.l1_tile.M_K_io_cycle_count * self.l1_tile_count
            self.K_N_l1_cycle_count = self.l1_tile.K_N_io_cycle_count * self.l1_tile_count

            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count()

        def simulate_l2_tile_io_cycle_count(
            self, access_type: L2AccessType, size_tuple: tuple[int, int], pcb_module: Device
        ): # cycles to load the tile from DRAM to l2
            mem_access_size = self.l2_status.access(access_type, self.coord_tuple, size_tuple)
            self.mem_access_size += mem_access_size
            mem_access_cycle = ceil(
                mem_access_size
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                ))
            return mem_access_cycle

        def simulate_l2_tile_compute_cycle_count(self):
            return self.l1_tile.compute_cycle_count * self.active_ctas_per_core

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            activation_dtype: DataType,
            weight_dtype: DataType,
            intermediate_dtype: DataType,
            pcb_module: Device,
            look_up_table: pd.DataFrame,
            is_2sm_mode: bool,
            dataflow: str,
        ):
            # performance counters
            self.systolic_array_fma_count = 0
            self.reg_access_count = 0
            self.l1_access_size = 0
            self.l2_access_size = 0

            self.M = M
            self.N = N
            self.K = K
            
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M // 2 if is_2sm_mode else M, N, K, activation_dtype, intermediate_dtype, dataflow, pcb_module, look_up_table
            )
            self.M_K_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M // 2 if is_2sm_mode else M, K, activation_dtype, pcb_module
            )
            self.K_N_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                K, N // 2 if is_2sm_mode else N, weight_dtype, pcb_module
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
            activation_dtype: DataType,
            intermediate_dtype: DataType,
            dataflow: str,
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            self.systolic_array_fma_count = M * N * K
            if activation_dtype.name == "fp16": # suppose HMMA.m16n8k16
                instruction_execution_count = self.systolic_array_fma_count // (16 * 8 * 16)
                self.reg_access_count = instruction_execution_count * (16 * 16 + 8 * 16) * activation_dtype.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_dtype.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_dtype.word_size // 4
            elif activation_dtype.name == "int8": # suppose HMMA.m16n8k32
                instruction_execution_count = self.systolic_array_fma_count // (16 * 8 * 32)
                self.reg_access_count = instruction_execution_count * (16 * 32 + 8 * 32) * activation_dtype.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_dtype.word_size // 4
                self.reg_access_count += instruction_execution_count * (16 * 8) * intermediate_dtype.word_size // 4
            elif activation_dtype.name == "fp8" or activation_dtype.name == "fp4":
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
                    dataflow,
                ))
                l1_access_size_temp = (M_tiling_factor * K * N + N_tiling_factor * M * K) * activation_dtype.word_size
                if compute_cycle_count_temp < compute_cycle_count:
                    compute_cycle_count = compute_cycle_count_temp
                    l1_access_size = l1_access_size_temp
                elif compute_cycle_count_temp == compute_cycle_count:
                    l1_access_size = min(l1_access_size, l1_access_size_temp)
            self.l1_access_size = l1_access_size

            return compute_cycle_count // (4 // activation_dtype.word_size)

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

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycle
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
