from __future__ import annotations

import torch
import sys

from utils import size, run_command, parse_nsys_stats, cleanup_files
from hardware_model.device import Device
from software_model.utils import data_type_dict, DataType, Tensor
from software_model.flashbase import FlashAttention
from math import ceil, isfinite, log2, floor
from scalesim.scale_sim import scalesim
from flash_attn import flash_attn_with_kvcache
import torch.cuda.profiler as profiler  
import copy

class FlashAttentionDecode(FlashAttention):
    """Analytical FlashDecoding simulator.

    The model follows FlashDecoding's algorithm: splitting the KV cache into chunks,
    processing them in parallel, and performing a final reduction.
    The simulator captures three critical contributions to latency:

    * Matrix multiply compute for QK^T and P@V on systolic arrays.
    * Vector-unit work for scaling, softmax (max/exp/sum), and dropout.
    * Global I/O traffic driven by repeatedly streaming K/V blocks and partial results.

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
        super().__init__(
            qkv_data_type,
            output_data_type,
            intermediate_data_type,
            pcb_module,
            is_causal,
            num_splits,
        )
        self.q_shape = None
        self.k_new_shape = None
        self.v_new_shape = None
        self.output_shape = None
        self.best_mapping = None
        self.look_up_table = None
        self.max_seqlen_kv = None
        
    def __call__(
        self,
        q: Tensor,
        k_new: Tensor,
        v_new: Tensor,
        cache_seqlens: int,
        max_seqlen_kv: int,
    ) -> Tensor:
        """Configures the operator for a FlashDecoding workload."""
        
        assert self.qkv_data_type == q.data_type == k_new.data_type == v_new.data_type
        
        assert len(q.shape) == len(k_new.shape) == len(v_new.shape) == 4, "q, k, v must have rank 4 [batch, heads, seq, dim]."
        
        assert max_seqlen_kv > 0, "max_seqlen_kv must be positive."
        
        for i in range(len(q.shape)):
            assert q.shape[i] > 0, "q shape dimensions must be positive."
            assert k_new.shape[i] == v_new.shape[i] > 0, "k, v shape must match and be positive."

        self.batch_size = q.shape[0]
        self.num_heads_q = q.shape[1]
        self.num_heads_kv = k_new.shape[1]
        self.max_seqlen_kv = max_seqlen_kv
        self.head_dim = q.shape[3]
        
        assert cache_seqlens > 0, "cache_seqlens length must be positive."
        
        self.cache_seqlens = cache_seqlens
        
        assert k_new.shape[2] == v_new.shape[2], "kv sequence lengths must match in decoding mode."
        
        k = Tensor([self.batch_size, self.num_heads_kv, k_new.shape[2] + cache_seqlens, self.head_dim], k_new.data_type)
        v = Tensor([self.batch_size, self.num_heads_kv, v_new.shape[2] + cache_seqlens, self.head_dim], v_new.data_type)
        
        self.seq_len_q = q.shape[2]
        self.seq_len_kv = k.shape[2]
        
        self.q_shape = q.shape
        self.k_new_shape = k_new.shape
        self.v_new_shape = v_new.shape
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
        
        self.global_reduction_flops = 0
        
        if self.num_splits == 0:
            self.num_splits = ceil(self.pcb_module.compute_module.core_count / self.total_heads)
        
        q_size = size([self.batch_size, self.num_heads_q, self.seq_len_q, self.head_dim])
        k_size = size([self.batch_size, self.num_heads_kv, self.seq_len_kv, self.head_dim])
        v_size = size([self.batch_size, self.num_heads_kv, self.seq_len_kv, self.head_dim])
        
        self.load_count = self.qkv_data_type.word_size * (q_size + k_size + v_size)
        self.store_count = self.output_data_type.word_size * size(self.output_shape)
        self.io_count = self.load_count + self.store_count
        
        if self.num_splits > 1:
            # Extra IO for reduction
            # We write num_splits partials, read them back.
            # Original store_count is for 1 output.
            # So we have (num_splits - 1) extra writes (since 1 is accounted for)
            # And num_splits reads.
            # Plus stats (l, m) which are 2 floats per head per seq_len_q per split.
            
            stats_size = self.total_heads * self.seq_len_q * self.num_splits * 2 * 4 # 4 bytes
            
            extra_store = (self.num_splits - 1) * self.store_count + stats_size
            extra_load = self.num_splits * self.store_count + stats_size
            
            self.global_reduction_flops = self.total_heads * self.seq_len_q * self.head_dim * self.num_splits * 5
            self.io_count += extra_store + extra_load
        
        self.computational_graph = self.ComputationGraph(
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_kv,
            head_dim=self.head_dim,
            batch_size=self.batch_size,
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_heads_kv,
            num_splits=self.num_splits,
            qkv_data_type=self.qkv_data_type,
            output_data_type=self.output_data_type,
            intermediate_data_type=self.intermediate_data_type,
        )

        return Tensor(self.output_shape, self.qkv_data_type)

    def roofline_model(self) -> float:
        self.roofline_latency = max(
            self.matmul_flop_count / (self.pcb_module.compute_module.get_total_systolic_array_throughput_per_cycle(self.qkv_data_type) * 2 * self.pcb_module.compute_module.clock_freq) + self.exp2_flops / (self.pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "exp2") * self.pcb_module.compute_module.clock_freq) + self.reduction_flops / (self.pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "reduction") * self.pcb_module.compute_module.clock_freq) + self.global_reduction_flops / (self.pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "fma") * self.pcb_module.compute_module.clock_freq),
            self.io_count 
            / min(self.pcb_module.io_module.bandwidth,
                self.pcb_module.compute_module.l2_bandwidth_per_cycle * self.pcb_module.compute_module.clock_freq
            ),
        )
        return self.roofline_latency

    def compile_and_simulate(
        self,
        compile_mode: str = "exhaustive",
    ):
        return super().compile_and_simulate(compile_mode=compile_mode)

    def run_on_gpu(self) -> float:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required to profile FlashDecoding.")
        if self.q_shape is None:
            raise RuntimeError("Call the operator with tensors before running on GPU.")
        
        target_args = [
            sys.executable, "-m",
            "test.FlashAttention.decode.decode_profile",
            "--batch_size", str(self.batch_size),
            "--nheads_q", str(self.num_heads_q),
            "--nheads_kv", str(self.num_heads_kv),
            "--headdim", str(self.head_dim),
            "--seqlen_q", str(self.seq_len_q),
            "--seqlen_kv", str(self.cache_seqlens),
            "--seqlen_kv_new", str(1),
            "--max_seqlen_kv", str(self.max_seqlen_kv),
            "--iterations", str(self.iterations),
            "--num_splits", "0",
            "--activation_dtype", self.qkv_data_type.name,
            "--weight_dtype", self.output_data_type.name,
        ]
        
        output, output_base = run_command(target_args)
        kernels_min_times = parse_nsys_stats(output)
        cleanup_files(output_base)
        
        print(f"FlashDecoding GPU kernels min times (ns): {kernels_min_times}")
        
        if kernels_min_times:
            total_ns = sum(kernels_min_times)
            self.latency_on_gpu = total_ns * 1e-9
        else:
            self.latency_on_gpu = 0
            
        return self.latency_on_gpu