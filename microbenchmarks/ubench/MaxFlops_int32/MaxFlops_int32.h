#ifndef MAXFLOPS_INT32_DEF_H
#define MAXFLOPS_INT32_DEF_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../../hw_def/hw_def.h"

#define REPEAT_ITERS 1024

// 测试 INT32 MAD 吞吐（mad.lo.s32）
// 结构参考你现在的 FP32 ILP 版本：4 个独立 accumulator 制造 ILP
__global__ void max_ops_int32_imad(uint32_t *startClk, uint32_t *stopClk,
                                   int32_t *data1, int32_t *data2, int32_t *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int32_t a = data1[gid];
  int32_t b = data2[gid];

  // 四个独立累加器，避免只有一条依赖链
  int32_t r0 = 0;
  int32_t r1 = 1;
  int32_t r2 = 2;
  int32_t r3 = 3;

  // 同步所有线程
  asm volatile("bar.sync 0;");

  // 记录起始 clock
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

#pragma unroll
  for (int j = 0; j < REPEAT_ITERS; ++j) {
    // 8 条 IMAD，每条 mad.lo.s32 视作 1 mul + 1 add = 2 ops
    // 4 个 accumulator，每个做两次 mad
    asm volatile(
        "{\n\t"
        "mad.lo.s32 %0, %4, %5, %0;\n\t"
        "mad.lo.s32 %1, %4, %5, %1;\n\t"
        "mad.lo.s32 %2, %4, %5, %2;\n\t"
        "mad.lo.s32 %3, %4, %5, %3;\n\t"
        "mad.lo.s32 %0, %4, %5, %0;\n\t"
        "mad.lo.s32 %1, %4, %5, %1;\n\t"
        "mad.lo.s32 %2, %4, %5, %2;\n\t"
        "mad.lo.s32 %3, %4, %5, %3;\n\t"
        "}\n"
        : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)
        : "r"(a), "r"(b));
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

// Host 端：计算 INT32 ops / clk / SM
float max_int32_flops() {
  intilizeDeviceProp(0);

  BLOCKS_NUM    = 1;  // 单 SM
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk  = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  int32_t  *data1    = (int32_t  *)malloc(TOTAL_THREADS * sizeof(int32_t));
  int32_t  *data2    = (int32_t  *)malloc(TOTAL_THREADS * sizeof(int32_t));
  int32_t  *res      = (int32_t  *)malloc(TOTAL_THREADS * sizeof(int32_t));

  uint32_t *startClk_g, *stopClk_g;
  int32_t  *data1_g, *data2_g, *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (int32_t)i;
    data2[i] = (int32_t)(i + 1);
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g,  TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g,    TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&data2_g,    TOTAL_THREADS * sizeof(int32_t)));
  gpuErrchk(cudaMalloc(&res_g,      TOTAL_THREADS * sizeof(int32_t)));

  gpuErrchk(cudaMemcpy(data1_g, data1,
                       TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2,
                       TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyHostToDevice));

  max_ops_int32_imad<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
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
                       TOTAL_THREADS * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));

  uint32_t start_min = startClk[0];
  uint32_t stop_max  = stopClk[0];
  for (int i = 1; i < TOTAL_THREADS; ++i) {
      if (startClk[i] < start_min) start_min = startClk[i];
      if (stopClk[i]  > stop_max)  stop_max  = stopClk[i];
  }

  double cycles = (double)(stop_max - start_min);

  // 每线程每次迭代：8 条 mad.lo.s32
  // 每条 mad 看成 1 mul + 1 add = 2 ops
  double mad_results_per_clk =
      (double)REPEAT_ITERS * (double)TOTAL_THREADS * 8.0 / cycles;
  double int32_ops_per_clk = 2.0 * mad_results_per_clk;

  printf("INT32 ops per SM per clk        = %.6f (ops/clk/SM)\n",
         (float)int32_ops_per_clk);
  printf("INT32 MAD results per SM per clk = %.6f (inst/clk/SM)\n",
         (float)mad_results_per_clk);
  printf("Total Clk number = %u\n", stopClk[0] - startClk[0]);

  // 你也可以像 FP32 那样，再除以 2 理解成“int32 ALUs per SM”
  return (float)int32_ops_per_clk;
}

#endif  // MAXFLOPS_INT32_DEF_H
