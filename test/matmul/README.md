# Before use the scripts in this folder

## Prepare and compile CUTLASS

Clone CUTLASS v4.3.2 with Thor support:

```
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout v4.3.2
```

Apply patch on https://github.com/NVIDIA/cutlass/issues/2784

Compile CUTLASS using the following command on Orin and Thor, respectively:
- Orin: 
```
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS='87' \
-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_*gemm*_tn* \
-DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j10
```
- Thor: 
```
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS='110a' \
-DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_tensorop_gemm_f8_f8_f32_void_*_tn*,cutlass3x_sm100_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_*_tn* \
-DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j10
```

Note that `tn` represents row-major x column-major. `bstensorop` represents block scaled tensor core GEMM for NVFP4.
