Code enhanced from Accel-sim.

Below are measured throughput statistics from microbenchmarks

| FMA (MAD) / SM / clk | A100 SM80         | Orin SM87          | 3090 SM86 |
|----------------------|-------------------|--------------------|-----------|
| INT32                | 64                | 64                 | 64        |
| FP16                 | `200 (256*80%)`   | `200 (256*80%)`    | 128       |
| FP32                 | 64                | 128                | 128       |
| FP64                 | 32                | 2                  | 2         |


Also see this link for theoretical throughput.
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions