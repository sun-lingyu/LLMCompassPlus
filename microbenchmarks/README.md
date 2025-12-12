Code enhanced from Accel-sim.

Before running the benchmark
- fix GPU freq to max 
`echo 1300500000 > /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq`
- fix mem freq to max
`echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked`
`echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state`
`echo 3199000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate`

To obtain memory bandwidth efficiency factor, seek to cuda_samples/Samples/1_Utilities/bandwidthTest. Compare its result with theoretic bandwidth.

Below are measured throughput statistics from microbenchmarks

| FMA (MAD) / SM / clk | A100 SM80         | Orin SM87          | 3090 SM86 |
|----------------------|-------------------|--------------------|-----------|
| INT32                | 64                | 64                 | 64        |
| FP16                 | `200 (256*80%)`   | `200 (256*80%)`    | 128       |
| FP32                 | 64                | 128                | 128       |
| FP64                 | 32                | 2                  | 2         |


Also see this link for theoretical throughput.
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions