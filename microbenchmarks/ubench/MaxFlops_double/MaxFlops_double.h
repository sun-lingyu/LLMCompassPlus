#ifndef MAXFLOPS_DOUBLE_DEF_H
#define MAXFLOPS_DOUBLE_DEF_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../hw_def/hw_def.h"

#define REPEAT_ITERS 1024

// 测试 FP64 FMA 吞吐（fma.rn.f64）
// 结构参考你现在的 FP32 版本：4 个独立 accumulator 制造 ILP
__global__ void max_flops_fp64_fma(uint32_t *startClk, uint32_t *stopClk,
                                   double *data1, double *data2, double *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  double a = data1[gid];
  double b = data2[gid];

  // 四个独立累加器，避免只有一条依赖链
  double r0 = 0.0;
  double r1 = 1.0;
  double r2 = 2.0;
  double r3 = 3.0;

  // 同步所有线程
  asm volatile("bar.sync 0;");

  // 记录起始 clock
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

#pragma unroll
  for (int j = 0; j < REPEAT_ITERS; ++j) {
    // 8 条 FMA，每条 fma.rn.f64 视作 2 FLOPs（1 mul + 1 add）
    asm volatile(
        "{\n\t"
        "fma.rn.f64 %0, %4, %5, %0;\n\t"
        "fma.rn.f64 %1, %4, %5, %1;\n\t"
        "fma.rn.f64 %2, %4, %5, %2;\n\t"
        "fma.rn.f64 %3, %4, %5, %3;\n\t"
        "fma.rn.f64 %0, %4, %5, %0;\n\t"
        "fma.rn.f64 %1, %4, %5, %1;\n\t"
        "fma.rn.f64 %2, %4, %5, %2;\n\t"
        "fma.rn.f64 %3, %4, %5, %3;\n\t"
        "}\n"
        : "+d"(r0), "+d"(r1), "+d"(r2), "+d"(r3)
        : "d"(a), "d"(b));
  }

  asm volatile("bar.sync 0;");

  // 记录结束 clock
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // 回写时间和结果，防止被优化掉
  startClk[gid] = start;
  stopClk[gid]  = stop;
  res[gid]      = r0 + r1 + r2 + r3;
}

// Host 端：计算 FP64 FMA 吞吐（FLOP/clk/SM）
float dpu_max_flops() {
  intilizeDeviceProp(0);

  BLOCKS_NUM    = 1;  // 单 SM
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk  = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  double   *data1    = (double   *)malloc(TOTAL_THREADS * sizeof(double));
  double   *data2    = (double   *)malloc(TOTAL_THREADS * sizeof(double));
  double   *res      = (double   *)malloc(TOTAL_THREADS * sizeof(double));

  uint32_t *startClk_g, *stopClk_g;
  double   *data1_g, *data2_g, *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (double)i;
    data2[i] = (double)(i + 1);
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g,  TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g,    TOTAL_THREADS * sizeof(double)));
  gpuErrchk(cudaMalloc(&data2_g,    TOTAL_THREADS * sizeof(double)));
  gpuErrchk(cudaMalloc(&res_g,      TOTAL_THREADS * sizeof(double)));

  gpuErrchk(cudaMemcpy(data1_g, data1,
                       TOTAL_THREADS * sizeof(double),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2,
                       TOTAL_THREADS * sizeof(double),
                       cudaMemcpyHostToDevice));

  max_flops_fp64_fma<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(startClk, startClk_g,
                       TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk,  stopClk_g,
                       TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res,      res_g,
                       TOTAL_THREADS * sizeof(double),
                       cudaMemcpyDeviceToHost));

  uint32_t start_min = startClk[0];
  uint32_t stop_max  = stopClk[0];
  for (int i = 1; i < TOTAL_THREADS; ++i) {
      if (startClk[i] < start_min) start_min = startClk[i];
      if (stopClk[i]  > stop_max)  stop_max  = stopClk[i];
  }

  double cycles = (double)(stop_max - start_min);

  // 每线程每次迭代：8 条 FMA
  // FMA 记作 2 FLOPs（1 mul + 1 add）
  double fma_results_per_clk =
      (double)REPEAT_ITERS * (double)TOTAL_THREADS * 8.0 / cycles;
  double flops_per_clk = 2.0 * fma_results_per_clk;

  printf("FP64 FLOPs per SM per clk          = %.6f (FLOP/clk/SM)\n",
         (float)flops_per_clk);
  printf("FP64 FMA results per SM per clk = %.6f (inst/clk/SM)\n",
         (float)fma_results_per_clk);
  printf("Total Clk number = %u\n", stopClk[0] - startClk[0]);

  // 释放内存
  cudaFree(startClk_g);
  cudaFree(stopClk_g);
  cudaFree(data1_g);
  cudaFree(data2_g);
  cudaFree(res_g);
  free(startClk);
  free(stopClk);
  free(data1);
  free(data2);
  free(res);

  return (float)flops_per_clk;
}

#endif  // MAXFLOPS_DOUBLE_DEF_H
