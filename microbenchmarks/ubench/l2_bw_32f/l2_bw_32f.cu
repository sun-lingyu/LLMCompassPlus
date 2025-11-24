// This code is a modification of L2 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
// https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the maximum read bandwidth of L2 cache for 32f
// Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../hw_def/hw_def.h"

#define REPEAT_TIMES 2048

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/
#define ILP 8  // 每个迭代发 8 个 load，确保 REPEAT_TIMES % ILP == 0

__global__ void l2_bw(uint64_t *startClk, uint64_t *stopClk, float *dsink,
                      float *posArray, unsigned ARRAY_SIZE) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;

  // 多路累加器，减弱依赖链
  float s0 = 0.f;
  float s1 = 0.f;
  float s2 = 0.f;
  float s3 = 0.f;
  float s4 = 0.f;
  float s5 = 0.f;
  float s6 = 0.f;
  float s7 = 0.f;

  // -------- warm up L2（沿用你原来的思路）--------
  for (uint32_t i = uid; i < ARRAY_SIZE; i += blockDim.x * gridDim.x) {
    float *ptr = posArray + i;
    asm volatile(
        "{\n\t"
        ".reg .f32 data;\n\t"
        "ld.global.cg.f32 data, [%1];\n\t"
        "add.f32 %0, data, %0;\n\t"
        "}\n"
        : "+f"(s0)
        : "l"(ptr)
        : "memory");
  }

  asm volatile("bar.sync 0;");

  // -------- start timing --------
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  const uint32_t iters = REPEAT_TIMES / ILP;

#pragma unroll
  for (uint32_t i = 0; i < iters; i++) {
    // 这一轮的 base index，保证 warp 内访问 coalesced
    uint32_t base = (i * ILP) * warpSize + uid;

    float *p0 = posArray + base + 0 * warpSize;
    float *p1 = posArray + base + 1 * warpSize;
    float *p2 = posArray + base + 2 * warpSize;
    float *p3 = posArray + base + 3 * warpSize;
    float *p4 = posArray + base + 4 * warpSize;
    float *p5 = posArray + base + 5 * warpSize;
    float *p6 = posArray + base + 6 * warpSize;
    float *p7 = posArray + base + 7 * warpSize;

    asm volatile(
        "{\n\t"
        ".reg .f32 d0,d1,d2,d3,d4,d5,d6,d7;\n\t"
        // 注意：%0-%7 是 s0-s7，%8-%15 是 p0-p7
        "ld.global.cg.f32 d0, [%8];\n\t"
        "ld.global.cg.f32 d1, [%9];\n\t"
        "ld.global.cg.f32 d2, [%10];\n\t"
        "ld.global.cg.f32 d3, [%11];\n\t"
        "ld.global.cg.f32 d4, [%12];\n\t"
        "ld.global.cg.f32 d5, [%13];\n\t"
        "ld.global.cg.f32 d6, [%14];\n\t"
        "ld.global.cg.f32 d7, [%15];\n\t"
        "add.f32 %0, d0, %0;\n\t"
        "add.f32 %1, d1, %1;\n\t"
        "add.f32 %2, d2, %2;\n\t"
        "add.f32 %3, d3, %3;\n\t"
        "add.f32 %4, d4, %4;\n\t"
        "add.f32 %5, d5, %5;\n\t"
        "add.f32 %6, d6, %6;\n\t"
        "add.f32 %7, d7, %7;\n\t"
        "}\n"
        : "+f"(s0), "+f"(s1), "+f"(s2), "+f"(s3),
          "+f"(s4), "+f"(s5), "+f"(s6), "+f"(s7)
        : "l"(p0), "l"(p1), "l"(p2), "l"(p3),
          "l"(p4), "l"(p5), "l"(p6), "l"(p7)
        : "memory");
  }

  asm volatile("bar.sync 0;");

  // -------- stop timing --------
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  startClk[bid * blockDim.x + tid] = start;
  stopClk[bid * blockDim.x + tid]  = stop;
  dsink[bid * blockDim.x + tid]    = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}

int main() {
  intilizeDeviceProp(0);

  unsigned ARRAY_SIZE = TOTAL_THREADS + REPEAT_TIMES * WARP_SIZE;
  assert(ARRAY_SIZE * sizeof(float) <
         L2_SIZE); // Array size must not exceed L2 size

  uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));

  float *posArray = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *dsink = (float *)malloc(TOTAL_THREADS * sizeof(float));

  float *posArray_g;
  float *dsink_g;
  uint64_t *startClk_g;
  uint64_t *stopClk_g;

  for (int i = 0; i < ARRAY_SIZE; i++)
    posArray[i] = (float)i;

  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&dsink_g, TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, ARRAY_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));

  l2_bw<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                           posArray_g, ARRAY_SIZE);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, TOTAL_THREADS * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float bw, BW;
  unsigned long long data =
      (unsigned long long)TOTAL_THREADS * REPEAT_TIMES * sizeof(float);

  uint64_t total_time = stopClk[0] - startClk[0];

  // uint64_t total_time =
  // *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS])-*std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
  bw = (float)(data) / ((float)(total_time));
  BW = bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "L2 bandwidth = " << bw << "(byte/clk), " << BW << "(GB/s)\n";
  float max_bw = L2_SLICES_NUM * L2_BANK_WIDTH_in_BYTE;
  BW = max_bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "Max Theortical L2 bandwidth = " << max_bw << "(byte/clk), "
            << BW << "(GB/s)\n";
  std::cout << "L2 BW achievable = " << (bw / max_bw) * 100 << "%\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return 1;
}
