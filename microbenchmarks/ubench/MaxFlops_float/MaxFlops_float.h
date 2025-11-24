#ifndef MAXFLOPS_FLOAT_DEF_H
#define MAXFLOPS_FLOAT_DEF_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../hw_def/hw_def.h"

#define REPEAT_ITERS 1024

// 这里专门测 FP32 FMA 吞吐
__global__ void max_flops_fp32_fma(uint32_t *startClk, uint32_t *stopClk,
                                   float *data1, float *data2, float *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  float a = data1[gid];
  float b = data2[gid];

  // 四个独立 accumulator，用来制造 ILP
  float r0 = 0.f;
  float r1 = 1.f;
  float r2 = 2.f;
  float r3 = 3.f;

  // 所有线程同步，避免 clock 被 warp 间 skew 污染
  asm volatile("bar.sync 0;");

  // 读 clock
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

#pragma unroll
  for (int j = 0; j < REPEAT_ITERS; ++j) {
    // 8 条 FMA，4 个 accumulator，每个做两次 FMA
    asm volatile(
        "{\n\t"
        "fma.rn.f32 %0, %4, %5, %0;\n\t"
        "fma.rn.f32 %1, %4, %5, %1;\n\t"
        "fma.rn.f32 %2, %4, %5, %2;\n\t"
        "fma.rn.f32 %3, %4, %5, %3;\n\t"
        "fma.rn.f32 %0, %4, %5, %0;\n\t"
        "fma.rn.f32 %1, %4, %5, %1;\n\t"
        "fma.rn.f32 %2, %4, %5, %2;\n\t"
        "fma.rn.f32 %3, %4, %5, %3;\n\t"
        "}\n"
        : "+f"(r0), "+f"(r1), "+f"(r2), "+f"(r3)
        : "f"(a), "f"(b));
  }

  asm volatile("bar.sync 0;");

  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  startClk[gid] = start;
  stopClk[gid]  = stop;
  // 防止被 DCE，做个归约
  res[gid] = r0 + r1 + r2 + r3;
}

float fpu_max_flops() {
  intilizeDeviceProp(0);

  BLOCKS_NUM   = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk  = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  float *data1       = (float *)malloc(TOTAL_THREADS * sizeof(float));
  float *data2       = (float *)malloc(TOTAL_THREADS * sizeof(float));
  float *res         = (float *)malloc(TOTAL_THREADS * sizeof(float));

  uint32_t *startClk_g, *stopClk_g;
  float *data1_g, *data2_g, *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (float)i;
    data2[i] = (float)(i + 1);
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g,  TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g,    TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&data2_g,    TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&res_g,      TOTAL_THREADS * sizeof(float)));

  gpuErrchk(cudaMemcpy(data1_g, data1,
                       TOTAL_THREADS * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2,
                       TOTAL_THREADS * sizeof(float),
                       cudaMemcpyHostToDevice));

  max_flops_fp32_fma<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g,
                       TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk,  stopClk_g,
                       TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res,      res_g,
                       TOTAL_THREADS * sizeof(float),
                       cudaMemcpyDeviceToHost));

  uint32_t start_min = startClk[0];
  uint32_t stop_max  = stopClk[0];
  for (int i = 1; i < TOTAL_THREADS; ++i) {
      if (startClk[i] < start_min) start_min = startClk[i];
      if (stopClk[i]  > stop_max)  stop_max  = stopClk[i];
  }

  double cycles = (double)(stop_max - start_min);

  // 每个线程：REPEAT_ITERS 次循环，每次 8 条 FMA
  // FMA 一般按 2 FLOPs 记（1 mul + 1 add）
  double fma_results_per_clk =
      (double)REPEAT_ITERS * (double)TOTAL_THREADS * 8.0 / cycles;
  double flops_per_clk = 2.0 * fma_results_per_clk;

  printf("FP32 FLOPs per SM per clk = %.6f (FLOP/clk/SM)\n",
         (float)flops_per_clk);
  printf("FP32 FMA results per SM per clk = %.6f (inst/clk/SM)\n",
         (float)fma_results_per_clk);
  printf("Total Clk number = %u\n", stopClk[0] - startClk[0]);

  return (float)flops_per_clk;
}

#endif
