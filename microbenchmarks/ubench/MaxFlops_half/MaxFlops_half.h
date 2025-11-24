#ifndef MAXFLOPS_FP16_HALF2_DEF_H
#define MAXFLOPS_FP16_HALF2_DEF_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../hw_def/hw_def.h"

#define REPEAT_ITERS 1024

// 小工具：half2 <-> uint32_t 的无开销 bitcast
union H2U32 {
  half2    h2;
  uint32_t u32;
};

// 测试 FP16 half2 FMA 吞吐（使用 inline PTX fma.rn.f16x2）
// 设计要点：
//   - 8 个独立 accumulator，增加 ILP
//   - 每个迭代 16 条 f16x2 FMA
//   - 计时窗口只覆盖 FMA loop，本身和你的 FP32/INT32 版本保持统一
__global__ void max_flops_fp16_half2(uint32_t *startClk, uint32_t *stopClk,
                                     __half *data1, __half *data2, __half *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // 构造 multiplier 向量 mult = (h1, h2)
  __half h1 = data1[gid];
  __half h2 = data2[gid];
  H2U32 m;
  m.h2 = __halves2half2(h1, h2);

  // 8 个 half2 accumulator，bitpattern 存在 8 个 uint32_t 里
  H2U32 t;
  t.h2 = __halves2half2(__float2half(0.0f), __float2half(1.0f));
  uint32_t r0 = t.u32;
  t.h2 = __halves2half2(__float2half(2.0f), __float2half(3.0f));
  uint32_t r1 = t.u32;
  t.h2 = __halves2half2(__float2half(4.0f), __float2half(5.0f));
  uint32_t r2 = t.u32;
  t.h2 = __halves2half2(__float2half(6.0f), __float2half(7.0f));
  uint32_t r3 = t.u32;
  t.h2 = __halves2half2(__float2half(8.0f), __float2half(9.0f));
  uint32_t r4 = t.u32;
  t.h2 = __halves2half2(__float2half(10.0f), __float2half(11.0f));
  uint32_t r5 = t.u32;
  t.h2 = __halves2half2(__float2half(12.0f), __float2half(13.0f));
  uint32_t r6 = t.u32;
  t.h2 = __halves2half2(__float2half(14.0f), __float2half(15.0f));
  uint32_t r7 = t.u32;

  // 同步所有线程
  asm volatile("bar.sync 0;");

  // 记录起始 clock
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

#pragma unroll
  for (int j = 0; j < REPEAT_ITERS; ++j) {
    // inline PTX：16 条 f16x2 FMA
    // fma.rn.f16x2 d, a, b, c  ==> d = a * b + c
    // 我们写成 d = mult * d + d，数值无所谓，关键是执行管线
    asm volatile(
        "{\n\t"
        "fma.rn.f16x2 %0, %8, %0, %0;\n\t"
        "fma.rn.f16x2 %1, %8, %1, %1;\n\t"
        "fma.rn.f16x2 %2, %8, %2, %2;\n\t"
        "fma.rn.f16x2 %3, %8, %3, %3;\n\t"
        "fma.rn.f16x2 %4, %8, %4, %4;\n\t"
        "fma.rn.f16x2 %5, %8, %5, %5;\n\t"
        "fma.rn.f16x2 %6, %8, %6, %6;\n\t"
        "fma.rn.f16x2 %7, %8, %7, %7;\n\t"
        "fma.rn.f16x2 %0, %8, %0, %0;\n\t"
        "fma.rn.f16x2 %1, %8, %1, %1;\n\t"
        "fma.rn.f16x2 %2, %8, %2, %2;\n\t"
        "fma.rn.f16x2 %3, %8, %3, %3;\n\t"
        "fma.rn.f16x2 %4, %8, %4, %4;\n\t"
        "fma.rn.f16x2 %5, %8, %5, %5;\n\t"
        "fma.rn.f16x2 %6, %8, %6, %6;\n\t"
        "fma.rn.f16x2 %7, %8, %7, %7;\n\t"
        "}\n"
        : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
          "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7)
        : "r"(m.u32));
  }

  asm volatile("bar.sync 0;");

  // 记录结束 clock
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // 写回时间
  startClk[gid] = start;
  stopClk[gid]  = stop;

  // 为了防止被优化掉：把 8 个 accumulator 合成一个结果写回
  H2U32 rr;
  __half sum = __float2half(0.0f);

  rr.u32 = r0;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r1;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r2;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r3;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r4;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r5;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r6;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  rr.u32 = r7;
  sum = __hadd(sum, __low2half(rr.h2));
  sum = __hadd(sum, __high2half(rr.h2));

  res[gid] = sum;
}

// Host 端：计算 half2 FMA 吞吐（FLOP/clk/SM）
float max_fp16_half2_flops() {
  intilizeDeviceProp(0);

  BLOCKS_NUM    = 1;  // 单 SM
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk  = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  __half   *data1    = (__half   *)malloc(TOTAL_THREADS * sizeof(__half));
  __half   *data2    = (__half   *)malloc(TOTAL_THREADS * sizeof(__half));
  __half   *res      = (__half   *)malloc(TOTAL_THREADS * sizeof(__half));

  uint32_t *startClk_g, *stopClk_g;
  __half   *data1_g, *data2_g, *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    float v = (float)i;
    data1[i] = __float2half(v);
    data2[i] = __float2half(v + 1.0f);
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g,  TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g,    TOTAL_THREADS * sizeof(__half)));
  gpuErrchk(cudaMalloc(&data2_g,    TOTAL_THREADS * sizeof(__half)));
  gpuErrchk(cudaMalloc(&res_g,      TOTAL_THREADS * sizeof(__half)));

  gpuErrchk(cudaMemcpy(data1_g, data1,
                       TOTAL_THREADS * sizeof(__half),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2,
                       TOTAL_THREADS * sizeof(__half),
                       cudaMemcpyHostToDevice));

  max_flops_fp16_half2<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
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
                       TOTAL_THREADS * sizeof(__half),
                       cudaMemcpyDeviceToHost));

  // 用整个 block 的最早开始和最晚结束作为计时窗口
  uint32_t start_min = startClk[0];
  uint32_t stop_max  = stopClk[0];
  for (int i = 1; i < TOTAL_THREADS; ++i) {
    if (startClk[i] < start_min) start_min = startClk[i];
    if (stopClk[i]  > stop_max)  stop_max  = stopClk[i];
  }

  double cycles = (double)(stop_max - start_min);

  // 每线程每迭代：16 条 f16x2 FMA
  // 一条 f16x2 FMA：2 lane × (mul+add) = 4 FLOPs
  double inst_per_clk =
      (double)REPEAT_ITERS * (double)TOTAL_THREADS * 16.0 / cycles;

  double results_per_clk = 2.0 * inst_per_clk;   // 每条 inst 有 2 个结果
  double flops_per_clk   = 4.0 * inst_per_clk;   // 每条 inst 有 4 FLOPs

  printf("Half2 FLOPs per SM per clk              = %.6f (FLOP/clk/SM)\n",
         (float)flops_per_clk);
  printf("Half2 FMA results per SM per clk        = %.6f (2xinst/clk/SM)\n",
         (float)results_per_clk);
  printf("Half2 FMA instructions per SM per clk   = %.6f (inst/clk/SM)\n",
         (float)inst_per_clk);
  printf("Total Clk number = %u\n", (unsigned)(stop_max - start_min));

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

#endif // MAXFLOPS_FP16_HALF2_DEF_H
