from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import os
import sys
import pandas as pd
import numpy as np

from utils import size, run_command, parse_nsys_stats, cleanup_files
from hardware_model.device import Device
from software_model.flashbase import FlashAttention
from software_model.utils import DataType, Tensor
from math import ceil, isfinite, log2
from scalesim.scale_sim import scalesim
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend
from flash_attn import flash_attn_func
import copy
    
class FlashAttentionPrefill(FlashAttention):
    def __init__(
        self,
        qkv_data_type: DataType,
        output_data_type: DataType,
        intermediate_data_type: DataType,
        pcb_module: Device,
        is_causal: bool = True,
    ) -> None:
        super().__init__(
            qkv_data_type=qkv_data_type,
            output_data_type=output_data_type,
            intermediate_data_type=intermediate_data_type,
            pcb_module=pcb_module,
            is_causal=is_causal,
            num_splits=1,
        )
        
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        """
        Configures the operator for a FlashAttention workload.
        
        Args:
            q (Tensor): Query tensor with shape [batch, heads, seq_len_q, head_dim].
            k (Tensor): Key tensor with shape [batch, heads, seq_len_kv, head_dim].
            v (Tensor): Value tensor with shape [batch, heads, seq_len_kv, head_dim].

        Returns:
            Tensor: Output tensor with shape [batch, heads, seq_len_q, head_dim].
        """
        
        assert self.qkv_data_type == q.data_type == k.data_type == v.data_type
        
        assert len(q.shape) == len(k.shape) == len(v.shape) == 4, "q, k, v must have rank 4 [batch, heads, seq, dim]."

        for i in range(len(q.shape)):
            assert q.shape[i] > 0, "q shape dimensions must be positive."
            assert k.shape[i] == v.shape[i] > 0, "k, v shape must match and be positive."

        self.batch_size = q.shape[0]
        self.num_heads_q = q.shape[1]
        self.num_heads_kv = k.shape[1]
        
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
        
        self.load_count = self.qkv_data_type.word_size * (q.size + k.size + v.size)
        self.store_count = self.qkv_data_type.word_size * size(self.output_shape)
        self.io_count = self.load_count + self.store_count
        
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

        return Tensor(self.output_shape, self.output_data_type)

    def roofline_model(self) -> float:
        """
        Estimates the latency using a roofline model.
        
        The latency is determined by the maximum of compute-bound latency and memory-bound latency.
        Compute-bound latency considers matrix multiplication, exponentiation, and reduction operations.
        Memory-bound latency considers I/O traffic and bandwidth.

        Args:
            pcb_module (Device): The hardware device configuration.

        Returns:
            float: Estimated latency in seconds.
        """
        self.roofline_latency = max(
            self.matmul_flop_count / (self.pcb_module.compute_module.get_total_systolic_array_throughput_per_cycle(self.qkv_data_type) * 2 * self.pcb_module.compute_module.clock_freq) + self.exp2_flops / (self.pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "exp2") * self.pcb_module.compute_module.clock_freq) + self.reduction_flops / (self.pcb_module.compute_module.get_total_vector_throughput_per_cycle(self.intermediate_data_type, "reduction") * self.pcb_module.compute_module.clock_freq),
            self.io_count 
            / min(self.pcb_module.io_module.bandwidth,
                self.pcb_module.compute_module.l2_bandwidth_per_cycle * self.pcb_module.compute_module.clock_freq
            ),
        )
        return self.roofline_latency

    def compile_and_simulate(
        self,
        compile_mode: str = "exhaustive",
    ) -> float:
        return super().compile_and_simulate(compile_mode=compile_mode)
    

    def run_on_gpu(self) -> float:
        """
        Runs the FlashAttention workload on a real GPU to measure latency.

        This is useful for validation and comparison with the analytical model.
        Requires a CUDA-enabled GPU and PyTorch with FlashAttention support.

        Returns:
            float: Measured latency in seconds.
        """
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required to profile FlashAttention.")
        if self.q_shape is None:
            raise RuntimeError("Call the operator with tensors before running on GPU.")
        
        target_args = [
            sys.executable, "-m",
            "test.FlashAttention.prefill.prefill_profile",
            "--batch_size", str(self.batch_size),
            "--nheads_q", str(self.num_heads_q),
            "--nheads_kv", str(self.num_heads_kv),
            "--head_dim", str(self.head_dim),
            "--seq_len", str(self.seq_len_q),
            "--iterations", str(self.iterations),
            "--activation_dtype", self.qkv_data_type.name,
            "--weight_dtype", self.output_data_type.name,
        ]
        
        output, output_base = run_command(target_args)
        kernels_min_time = parse_nsys_stats(output)
        cleanup_files(output_base)
        
        if kernels_min_time:
            total_ns = sum(kernels_min_time)
            self.latency_on_gpu = total_ns * 1e-9
        else:
            self.latency_on_gpu = 0
        
        return self.latency_on_gpu