from utils import size
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType, DeviceType, L2AccessType, L2Cache, data_type_dict
from math import ceil, inf
from enum import Enum
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
            print(f"swizzle_size: {self.swizzle_size}, raster_order: {'ALONG_M' if self.raster_order == Matmul.RasterOrder.ALONG_M else 'ALONG_N'}")
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
                    scale_dtype: DataType = data_type_dict["fp8"], # NVFP4 use fp8 (ue4m3) scale
                    L2Cache_previous: L2Cache = None
                    ):
            super().__init__(l2_size)
            # change output to activation
            assert M > 0 and N > 0 and K > 0
            scale_block_size = activation_dtype.scale_block_size

            self.m_tiles = ceil(M / L2Cache.TILE_LENGTH)
            self.n_tiles = ceil(N / L2Cache.TILE_LENGTH)
            self.k_tiles = ceil(K / L2Cache.TILE_LENGTH)
            self.n_scale_tiles = ceil(N / L2Cache.TILE_LENGTH / scale_block_size) if scale_block_size else None
            self.k_scale_tiles = ceil(K / L2Cache.TILE_LENGTH / scale_block_size) if scale_block_size else None
            self.activation_tile_size = activation_dtype.word_size * self.TILE_LENGTH ** 2
            self.weight_tile_size = weight_dtype.word_size * self.TILE_LENGTH ** 2
            self.output_tile_size = output_dtype.word_size * self.TILE_LENGTH ** 2
            self.scale_tile_size = scale_dtype.word_size * self.TILE_LENGTH ** 2

            if L2Cache_previous:
                assert L2Cache_previous.output_tile_size == self.activation_tile_size
                while L2Cache_previous.resident_tiles:
                    tile = L2Cache_previous.resident_tiles.popitem(last=False)[0]
                    if tile.access_type == L2AccessType.OUTPUT:
                        self.resident_tiles[L2Cache.Tile(L2AccessType.ACTIVATION, tile.coord_tuple)] = None
                        self.occupied_size += L2Cache_previous.output_tile_size
                    if tile.access_type == L2AccessType.OUTPUT_SCALE:
                        self.resident_tiles[L2Cache.Tile(L2AccessType.ACTIVATION_SCALE, tile.coord_tuple)] = None
                        self.occupied_size += L2Cache_previous.scale_tile_size
        
        def access(self,
                   access_type: L2AccessType,
                   coord_tuple: tuple[int, int],
                   scope_tuple: tuple[int, int]
                   ):
            height = self.m_tiles if access_type == L2AccessType.ACTIVATION else \
                    self.k_tiles if access_type == L2AccessType.WEIGHT else \
                    self.m_tiles if access_type == L2AccessType.OUTPUT else \
                    self.m_tiles if access_type == L2AccessType.ACTIVATION_SCALE else \
                    self.k_scale_tiles if access_type == L2AccessType.WEIGHT_SCALE else \
                    self.m_tiles if access_type == L2AccessType.OUTPUT_SCALE else \
                    None
            width = self.k_tiles if access_type == L2AccessType.ACTIVATION else \
                    self.n_tiles if access_type == L2AccessType.WEIGHT else \
                    self.n_tiles if access_type == L2AccessType.OUTPUT else \
                    self.k_scale_tiles if access_type == L2AccessType.ACTIVATION_SCALE else \
                    self.n_tiles if access_type == L2AccessType.WEIGHT_SCALE else \
                    self.n_scale_tiles if access_type == L2AccessType.OUTPUT_SCALE else \
                    None
            tile_size = self.activation_tile_size if access_type == L2AccessType.ACTIVATION else \
                    self.weight_tile_size if access_type == L2AccessType.WEIGHT else \
                    self.output_tile_size if access_type == L2AccessType.OUTPUT else \
                    self.scale_tile_size
            assert height and width
            assert coord_tuple[0] % L2Cache.TILE_LENGTH == 0 and coord_tuple[1] % L2Cache.TILE_LENGTH == 0
            assert scope_tuple[0] % L2Cache.TILE_LENGTH == 0 and scope_tuple[1] % L2Cache.TILE_LENGTH == 0
            assert coord_tuple[0] >= 0 and coord_tuple[0] + scope_tuple[0] <= height * L2Cache.TILE_LENGTH, f"coord_tuple[0]: {coord_tuple[0]}, scope_tuple[0]: {scope_tuple[0]}, height * L2Cache.TILE_LENGTH: {height * L2Cache.TILE_LENGTH}"
            assert coord_tuple[1] >= 0 and coord_tuple[1] + scope_tuple[1] <= width * L2Cache.TILE_LENGTH, f"coord_tuple[1]: {coord_tuple[1]}, scope_tuple[1]: {scope_tuple[1]}, width * L2Cache.TILE_LENGTH: {width * L2Cache.TILE_LENGTH}"

            mem_access_size = 0
            for i in range(coord_tuple[0], coord_tuple[0] + scope_tuple[0], L2Cache.TILE_LENGTH):
                for j in range(coord_tuple[1], coord_tuple[1] + scope_tuple[1], L2Cache.TILE_LENGTH):
                    tile = self.Tile(access_type, (i, j))
                    if tile in self.resident_tiles: # HIT
                        # assert access_type not in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE)
                        self.resident_tiles.move_to_end(tile) # update LRU
                    else:
                        while self.occupied_size + tile_size > self.l2_size: # EVICT
                            mem_access_size += self.evict_oldest_tile()
                        if access_type not in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE): # load from DRAM
                            mem_access_size += tile_size
                        self.occupied_size += tile_size
                        self.resident_tiles[self.Tile(access_type, (i, j))] = None
            self.total_mem_access_size += mem_access_size
            return mem_access_size

        def evict_oldest_tile(self):
            assert self.resident_tiles
            
            mem_access_size = 0
            oldest_tile = self.resident_tiles.popitem(last=False)[0]
            tile_size = self.activation_tile_size if oldest_tile.access_type == L2AccessType.ACTIVATION else \
                    self.weight_tile_size if oldest_tile.access_type == L2AccessType.WEIGHT else \
                    self.output_tile_size if oldest_tile.access_type == L2AccessType.OUTPUT else \
                    self.scale_tile_size
            if oldest_tile.access_type in (L2AccessType.OUTPUT, L2AccessType.OUTPUT_SCALE):
                mem_access_size += tile_size
            self.occupied_size -= tile_size
            self.total_mem_access_size += mem_access_size
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
        
        assert device in ["Orin", "Thor"], "Only support Orin and Thor!"
        self.device_type = DeviceType.ORIN if device =="Orin" else DeviceType.THOR
        assert (self.device_type == DeviceType.ORIN and weight_dtype.name in ("fp16", "int8", "int4")) \
            or (self.device_type == DeviceType.THOR and weight_dtype.name in ("fp8", "fp4")), \
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
        if self.activation_dtype.name == "fp4":
            self.io_count += int(data_type_dict["fp8"].word_size * (self.M * self.K + self.K * self.N  + self.M * self.N) / self.activation_dtype.scale_block_size)
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
                * pcb_module.compute_module.clock_freq
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
        assert (self.device_type == DeviceType.THOR and M >= 64) \
            or (self.device_type == DeviceType.ORIN and M >= 32), \
            "at least 64/32 to fit Thor/Orin GEMM workflow"
        assert K >= 256, "at least 256 to accurately model loop-K"
        
        if self.weight_dtype.name in ("fp16", "int8"):
            assert self.device_type == DeviceType.ORIN
            cta_m_list = [64, 128, 256]
            cta_n_list = [64, 128, 256]
            cta_k_list = [32, 64]
        elif self.weight_dtype.name == "int4": # Marlin
            cta_m_list = [32]
            cta_n_list = [256]
            cta_k_list = [64]
        elif self.weight_dtype.name == "fp8":
            assert self.device_type == DeviceType.THOR
            cta_m_list = [64, 128, 256]
            cta_n_list = [64, 128, 256]
            cta_k_list = [128]
        elif self.weight_dtype.name == "fp4":
            assert self.device_type == DeviceType.THOR
            cta_m_list = [128, 256]
            cta_n_list = [64, 128, 256] # discard 192 since it may not be divisible by N
            cta_k_list = [256]
        else:
            assert False
        
        for cta_m in cta_m_list:
            for cta_n in cta_n_list:
                for cta_k in cta_k_list:
                    for swizzle_size in [1, 2, 4, 8]:
        # for cta_m in [256]:
        #     for cta_n in [256]:
        #         for cta_k in [256]:
                    # for swizzle_size in [1]:
                        activation_tile_size = int(cta_m * cta_k * self.activation_dtype.word_size)
                        weight_tile_size = int(cta_n * cta_k * self.weight_dtype.word_size)
                        output_tile_size = int(cta_m * cta_n * self.output_dtype.word_size)
                        intermediate_tile_size = int(cta_m * cta_n * self.intermediate_dtype.word_size)
                        if self.device_type == DeviceType.ORIN:
                            is_2sm_mode = False
                        elif self.device_type == DeviceType.THOR:
                            if self.activation_dtype.name == "fp8" and cta_m in (128, 256):
                                is_2sm_mode = True
                            elif self.activation_dtype.name == "fp4" and cta_m == 256:
                                is_2sm_mode = True
                            else:
                                is_2sm_mode = False
                        else:
                            assert False
                        l1_working_set_size = (activation_tile_size + weight_tile_size)
                        if is_2sm_mode:
                            l1_working_set_size //= 2
                        
                        if cta_m >= M * 2:
                            continue

                        # Check enough stages
                        if self.device_type == DeviceType.THOR:
                            # Blackwell: CUTLASS StageCountAuto Policy: 1 CTA/SM
                            stages = pcb_module.compute_module.core.SRAM_size // l1_working_set_size
                            if stages <= 3:
                                continue # not enough pipeline stages
                        elif self.device_type == DeviceType.ORIN:
                            # Ampere: 3-5 stages to maximize active_ctas_per_core
                            stages = 4
                        else:
                            assert False

                        # Check l1 usage
                        l1_working_set_size *= stages
                        if l1_working_set_size > pcb_module.compute_module.core.SRAM_size:
                            continue

                        # Check TMEM usage
                        if self.device_type == DeviceType.THOR:
                            intermediate_size = intermediate_tile_size
                            if is_2sm_mode:
                                intermediate_size //= 2
                            assert intermediate_size <= 256 * 1024 # TMEM: 256KB

                        # Check Register usage
                        register_usage = 1
                        if self.device_type == DeviceType.ORIN:
                            if self.activation_dtype.name == "fp16": # suppose HMMA.m16n8k16
                                register_usage = cta_m * cta_n * self.intermediate_dtype.word_size // 4 + \
                                    (16 * 16 + 16 * 8) * self.activation_dtype.word_size // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                            elif self.activation_dtype.name == "int8": # suppose HMMA.m16n8k32
                                register_usage = cta_m * cta_n * self.intermediate_dtype.word_size // 4 + \
                                    (16 * 32 + 32 * 8) * self.activation_dtype.word_size // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                            else:
                                assert False, "not implemented yet"
                            if self.weight_dtype.name == "int4":
                                register_usage += (16 * 8) * (self.activation_dtype.word_size + self.weight_dtype.word_size) // 4 * pcb_module.compute_module.core.sublane_count * 2 # double buffering
                            if register_usage > pcb_module.compute_module.core.total_registers:
                                continue        
                        
                        # print(f"cta_m: {cta_m}, cta_n: {cta_n}, cta_k: {cta_k}")
                        # print(f"l1_working_set_size: {l1_working_set_size / 1024} KB, register_usage: {register_usage}")
                        active_ctas_per_core = int(min(
                            pcb_module.compute_module.core.SRAM_size // l1_working_set_size, 
                            pcb_module.compute_module.core.total_registers // register_usage
                        ))
                        if self.device_type == DeviceType.THOR:
                            assert active_ctas_per_core == 1
                        
                        raster_order = Matmul.RasterOrder.ALONG_M if M <= N else Matmul.RasterOrder.ALONG_N
                        mapping = self.Mapping(
                            cta_m,
                            cta_n,
                            cta_k,
                            stages,
                            swizzle_size,
                            raster_order,
                            is_2sm_mode,
                            active_ctas_per_core
                        )
                        l2_status = Matmul.L2CacheMatmul(
                            pcb_module.compute_module.l2_size, 
                            M,
                            N,
                            K,
                            self.activation_dtype,
                            self.weight_dtype,
                            self.output_dtype,
                        )
                        pending_write_cycle = 0
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
                        # print(f"pending_write_cycle: {pending_write_cycle}")
                        # print(f"drain_cycle: {drain_cycle}")
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
        swizzle_size = mapping.swizzle_size if mapping.swizzle_size > 1 else ceil(M / cta_m) # 1 means no swizzle
        is_2sm_mode = mapping.is_2sm_mode
        dataflow = mapping.dataflow
        total_ctas = ceil(M / cta_m) * ceil(N / cta_n)

        cta_sequence = []
        # L2 swizzling
        if raster_order == Matmul.RasterOrder.ALONG_M:
            for m_base in range(0, ceil(M / cta_m), swizzle_size):
                for n in range(ceil(N / cta_n)):
                    for m_offset in range(swizzle_size):
                        m = m_base + m_offset
                        if m < ceil(M / cta_m):
                            cta_sequence.append((m * cta_m, n * cta_n))
        elif raster_order == Matmul.RasterOrder.ALONG_N:
            for n_base in range(0, ceil(N / cta_n), swizzle_size):
                for m in range(ceil(M / cta_m)):
                    for n_offset in range(swizzle_size):
                        n = n_base + n_offset
                        if n < ceil(N / cta_n):
                            cta_sequence.append((m * cta_m, n * cta_n))

        # Build waves
        shared_l1_simulator_args = (
            self.activation_dtype, self.weight_dtype, self.intermediate_dtype, self.output_dtype,
            pcb_module, self.look_up_table, is_2sm_mode, dataflow
        )
        if K // cta_k > 0:
            normal_l1_tile = Matmul.L1TileSimulator(cta_m, cta_n, cta_k, *shared_l1_simulator_args)
        if K % cta_k > 0:
            tail_l1_tile = Matmul.L1TileSimulator(cta_m, cta_n, K % cta_k, *shared_l1_simulator_args)
        waves = []
        for i in range(0, total_ctas, active_ctas_per_wave):
            cta_chunk = cta_sequence[i : i + active_ctas_per_wave]
            curr_wave = []
            m_coords = [p[0] for p in cta_chunk]
            n_coords = [p[1] for p in cta_chunk]
            for k_start in range(0, K, cta_k):
                k_size = min(cta_k, K - k_start)
                curr_wave.append(
                    self.L2TileSimulator(
                        max(m_coords) - min(m_coords) + cta_m, max(n_coords) - min(n_coords) + cta_n, k_size,
                        min(m_coords), min(n_coords), k_start, cta_chunk,
                        tail_l1_tile if k_size < cta_k else normal_l1_tile,
                        ceil(len(cta_chunk) / execution_unit_num), l2_status, self.activation_dtype, pcb_module
                    )
                )
            waves.append(curr_wave)

        # ----------------Begin Counting cycles-------------------
        total_cycle_count = 0
        for l2_tiles in waves:
            # Prologue
            input_io_cycle_count = l2_tiles[0].get_input_io_cycle_count()
            total_cycle_count += input_io_cycle_count
            if input_io_cycle_count == l2_tiles[0].l1_input_io_cycle_count: # Not accessing DRAM
                pending_write_cycle = max(0, pending_write_cycle - input_io_cycle_count)
            else:
                total_cycle_count += pending_write_cycle
                pending_write_cycle = 0
            if self.weight_dtype.name == "int4":
                offset_for_w4a16 = 0.03 # obtained by fitting real machine cycles
                total_cycle_count += l2_tiles[0].K * l2_tiles[0].N * offset_for_w4a16 # mainly models non-overlapped dequant overhead

            # Loop K: double buffering
            wait_ready_cycle = 0
            for iter in range(ceil(K / cta_k)):
                # current tile compute latency
                curr_iter_cycle_count = wait_ready_cycle + l2_tiles[iter].compute_cycle_count
                total_cycle_count += curr_iter_cycle_count
                if input_io_cycle_count == l2_tiles[iter].l1_input_io_cycle_count: # Not accessing DRAM
                    pending_write_cycle = max(0, pending_write_cycle - curr_iter_cycle_count)
                else:
                    total_cycle_count += pending_write_cycle
                    pending_write_cycle = 0
                
                # print(f"wait_ready_cycle {wait_ready_cycle}, l2_tiles[iter].compute_cycle_count {l2_tiles[iter].compute_cycle_count}")

                # update wait_ready_cycle
                if iter + 1 < ceil(K / cta_k):
                    input_io_cycle_count = l2_tiles[iter + 1].get_input_io_cycle_count()
                    wait_ready_cycle = max(0, input_io_cycle_count - l2_tiles[iter].compute_cycle_count)
                else:
                    wait_ready_cycle = 0

            # Epilogue
            total_cycle_count += l2_tiles[0].M * l2_tiles[0].N // pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.output_dtype, "cvt")
            output_io_cycle_count = l2_tiles[-1].get_output_io_cycle_count()
            if self.device_type == DeviceType.ORIN:
                offset_for_smem_reorganizing_etc = 0.015 # obtained by fitting real machine cycles
                total_cycle_count += l2_tiles[-1].M * l2_tiles[-1].N * offset_for_smem_reorganizing_etc # offset, mainly models data reorganizing through smem before write to DRAM. For Blackwell, smem data reorganizing is taken over asynchronously by TMA hardware.
            pending_write_cycle += output_io_cycle_count
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
            coord_K: int,
            cta_chunk: list,
            l1_tile: "Matmul.L1TileSimulator",
            active_ctas_per_core: int,
            l2_status: "Matmul.L2CacheMatmul",
            activation_dtype: DataType,
            pcb_module: Device
        ):
            self.M = M
            self.N = N
            self.K = K
            self.coord_M = coord_M
            self.coord_N = coord_N
            self.coord_K = coord_K
            self.cta_chunk = cta_chunk
            self.l1_tile = l1_tile
            self.l2_status = l2_status
            self.activation_dtype = activation_dtype
            self.pcb_module = pcb_module

            self.l1_input_io_cycle_count = self.l1_tile.input_io_cycle_count * len(self.cta_chunk)
            self.l1_output_io_cycle_count = self.l1_tile.output_io_cycle_count * len(self.cta_chunk)
            self.l1_compute_cycle_count = self.l1_tile.compute_cycle_count
            self.compute_cycle_count = self.l1_compute_cycle_count * active_ctas_per_core
        
        def get_input_io_cycle_count(self):
            M_K_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                L2AccessType.ACTIVATION, 
                (self.coord_M, self.coord_K),
                (self.M, self.K),
                self.pcb_module
            )
            K_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                L2AccessType.WEIGHT, 
                (self.coord_K, self.coord_N),
                (self.K, self.N),
                self.pcb_module
            )
            if self.activation_dtype.name == "fp4":
                scale_block_size = self.activation_dtype.scale_block_size
                M_K_io_cycle_count += self.simulate_l2_tile_io_cycle_count(
                    L2AccessType.ACTIVATION_SCALE, 
                    (self.coord_M, (self.coord_K // scale_block_size) // L2Cache.TILE_LENGTH * L2Cache.TILE_LENGTH),
                    (self.M, ceil((self.K // scale_block_size) / L2Cache.TILE_LENGTH) * L2Cache.TILE_LENGTH),
                    self.pcb_module
                )
                K_N_io_cycle_count += self.simulate_l2_tile_io_cycle_count(
                    L2AccessType.WEIGHT_SCALE, 
                    ((self.coord_K // scale_block_size) // L2Cache.TILE_LENGTH * L2Cache.TILE_LENGTH, self.coord_N),
                    (ceil((self.K // scale_block_size) / L2Cache.TILE_LENGTH) * L2Cache.TILE_LENGTH, self.N),
                    self.pcb_module
                )
            return max(
                M_K_io_cycle_count + K_N_io_cycle_count,
                self.l1_input_io_cycle_count
            )
        
        def get_output_io_cycle_count(self):
            M_N_io_cycle_count = 0
            for cta_coord in self.cta_chunk: # May not be a rectangle
                M_N_io_cycle_count += self.simulate_l2_tile_io_cycle_count(
                    L2AccessType.OUTPUT, 
                    cta_coord,
                    (self.l1_tile.M, self.l1_tile.N),
                    self.pcb_module
                )
                if self.activation_dtype.name == "fp4":
                    scale_block_size = self.activation_dtype.scale_block_size
                    M_N_io_cycle_count += self.simulate_l2_tile_io_cycle_count(
                        L2AccessType.OUTPUT_SCALE,
                        (cta_coord[0], (cta_coord[1] // scale_block_size) // L2Cache.TILE_LENGTH * L2Cache.TILE_LENGTH),
                        (self.l1_tile.M, ceil((self.l1_tile.N // scale_block_size) / L2Cache.TILE_LENGTH) * L2Cache.TILE_LENGTH),
                        self.pcb_module
                    ) # Round access granularity to L2Cache.TILE_LENGTH. This may result in OUTPUT_SCALE HIT in L2, which is ok.
            return max(
                M_N_io_cycle_count,
                self.l1_output_io_cycle_count
            )

        def simulate_l2_tile_io_cycle_count(
            self, access_type: L2AccessType, coord_tuple: tuple[int, int], size_tuple: tuple[int, int], pcb_module: Device
        ): # cycles to load the tile from DRAM to l2
            mem_access_size = self.l2_status.access(access_type, coord_tuple, size_tuple)
            mem_access_cycle = ceil(
                mem_access_size
                / (
                    pcb_module.io_module.bandwidth
                    * pcb_module.io_module.bandwidth_efficiency
                    / pcb_module.compute_module.clock_freq
                ))
            return mem_access_cycle
            
    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            activation_dtype: DataType,
            weight_dtype: DataType,
            intermediate_dtype: DataType,
            output_dtype: DataType,
            pcb_module: Device,
            look_up_table: pd.DataFrame,
            is_2sm_mode: bool,
            dataflow: str
        ):
            # performance counters
            self.M = M
            self.N = N
            self.K = K
            self.is_2sm_mode = is_2sm_mode
            self.activation_dtype = activation_dtype
            self.weight_dtype = weight_dtype
            self.intermediate_dtype = intermediate_dtype
            self.output_dtype = output_dtype
            self.pcb_module = pcb_module
            self.look_up_table = look_up_table
            self.dataflow = dataflow

            M_K_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                self.M // 2 if self.is_2sm_mode else self.M, self.K, self.activation_dtype, self.pcb_module
            )
            K_N_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                self.K, self.N // 2 if self.is_2sm_mode else self.N, self.weight_dtype, self.pcb_module
            )
            M_N_io_cycle_count = self.simulate_l1_tile_io_cycle_count(
                self.M // 2 if self.is_2sm_mode else self.M, self.N, self.output_dtype, self.pcb_module
            )
            self.input_io_cycle_count = M_K_io_cycle_count + K_N_io_cycle_count
            self.output_io_cycle_count = M_N_io_cycle_count
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                self.M // 2 if self.is_2sm_mode else self.M, self.N, self.K, self.activation_dtype, self.intermediate_dtype, self.dataflow, self.pcb_module, self.look_up_table
            )
        
        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ): # cycles to load the tile from L2 to L1
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
            assert M >= pcb_module.compute_module.core.systolic_array.array_width
            assert N >= pcb_module.compute_module.core.systolic_array.array_height
            assert K >= 32
            # compute_cycle_count = inf
            # for (
            #     M_tiling_factor,
            #     N_tiling_factor, # does not model sliced-K
            # ) in Matmul.all_factor_pairs(pcb_module.compute_module.core.sublane_count):
            #     compute_cycle_count_temp = ceil(Matmul.simulate_systolic_array_cycle_count(
            #         look_up_table,
            #         ceil(M / M_tiling_factor),
            #         ceil(N / N_tiling_factor),
            #         K,
            #         pcb_module.compute_module.core.systolic_array.array_height,
            #         pcb_module.compute_module.core.systolic_array.array_width,
            #         dataflow,
            #     ))
            compute_cycle_count = M* N * K // (pcb_module.compute_module.core.systolic_array.array_height * pcb_module.compute_module.core.systolic_array.array_width * pcb_module.compute_module.core.sublane_count)
            
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
        return cycle_count
