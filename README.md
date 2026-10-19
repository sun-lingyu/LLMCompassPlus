[![DOI](https://zenodo.org/badge/779008229.svg)](https://zenodo.org/doi/10.5281/zenodo.10892431)

# LLMCompass

This repository provides the implementation of **LLMCompass** from the following papers:

[**LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference**](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf)

*Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, David Wentzlaff*


In the Proceedings of the 51st Annual International Symposium on Computer Architecture:

```
@inproceedings{LLMCompass,
author = {Zhang, Hengrui and Ning, August and Prabhakar, Rohan Baskar and Wentzlaff, David},
title = {LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference},
year = {2024},
booktitle = {Proceedings of the 51st Annual International Symposium on Computer Architecture},
}
```


## Set up the environment

```
$ conda create -n llmcompass_ae python=3.9
$ conda activate llmcompass_ae
$ pip3 install scalesim
$ conda install pytorch==2.0.0 -c pytorch
$ pip3 install matplotlib
$ pip3 install seaborn
$ pip3 install scipy
```

## Installation

### If using Github
```
$ git clone -b ISCA_AE https://github.com/PrincetonUniversity/LLMCompass
$ cd LLMCompass
$ git submodule init
$ git submodule update --recursive
```

### If using Zenodo
Unzip the file and download from https://github.com/PrincetonUniversity/ttm-cas.git as `cost_model\supply_chain`


### If using Docker
A Dockerfile has been provided (`./Dockerfile`), including all the software dependencies and the LLMCompass source code.

A docker image has been provided [here](https://github.com/HenryChang213/LLMCompass_ISCA_AE_docker).

## AE Experiment workflow
```
# Figure 5 (around 100 min) 
$ cd ae/figure5
$ bash run_figure5.sh 

# Figure 6 (around 1 min)
$ cd ae/figure6
$ bash run_figure6.sh

# Figure 7 (around 20 min)
$ cd ae/figure7
$ bash run_figure7.sh

# Figure 8 (around 40 min)
$ cd ae/figure8
$ bash run_figure8.sh

# Figure 9 (around 30 min)
$ cd ae/figure9
$ bash run_figure9.sh

# Figure 10 (around 45 min)
$ cd ae/figure10
$ bash run_figure10.sh

# Figure 11 (around 5 min) 
$ cd ae/figure11
$ bash run_figure11.sh

# Figure 12 (around 4 hours) 
$ cd ae/figure12
$ bash run_figure12.sh
```

## AE Expected result

After running each script above, the corresponding figures
will be generated under the corresponding directory as suggested by its name.

For comparison, a copy of the expected results can be found in `ae\expected_results`


## User Guide

A guide on "How to Run a LLMCompass Simulation" is shown [here](./docs/run.md).

## Cross-operator L2 write-back (2026-10-09)

The shared L2 model retains physical tile identity, LRU order and dirty state
across Matmul, Quant, SiLU, residual/norm, in-place QK/RoPE, FlashAttention and
decode KV append. Each mapping candidate gets a cache snapshot; evaluating it
cannot mutate its predecessor. Evicted dirty tiles incur DRAM traffic at the
access causing eviction. FP4 data and scale tiles retain their own byte sizes.

Correlation/standalone callers can use `compile_and_simulate(...,
drain_at_end=True)` to include final dirty output writes. The default operator
accounting serializes this drain after execution. Elementwise-like operators
instead take the maximum of compute time, the complete DRAM stream (including
drain), and L2 service time; their raw drain cycles are not additive latency.
Block calls in `simulate` use `drain_at_end=False`, preserving dirty data for
successors. This does not suppress capacity-eviction writes. Operator attributes
`execution_cycles`, `drain_cycles`, `drain_bytes` and `cache_metrics` expose the
accounting. A drain cleans resident tiles without invalidating them by default.

Block cache format is version 7; older cached layer results are not reused.
The tile model remains fully associative and uses 16x16 allocation granularity.
The standalone drain model is not a cycle-exact model of profiler pipelining.

## Communication modes (2026-10-19)

`python -m simulate.main` accepts `--comm_mode serial|ideal` (default `serial`):

- `serial` supports TP only. Independent Ring AllGather and ReduceScatter
  operators execute between compute operators and retain the full L2 cache
  state, including LRU order and dirty tiles. AG expands the M dimension from
  M/TP to M; RS shrinks it back. M/TP must be divisible by 16.
- `ideal` supports TP and prefill CP. Communication takes zero time, and
  reported latency and power cover compute only. TP retains the same AG/RS
  layout and cache transitions as `serial`; CP retains its AllToAll cache
  boundaries. Bandwidth and communication startup parameters are ignored.

`--comm_bandwidth` is one physical link's one-way bandwidth in GB/s. TP=2 uses
X and TP=4 uses aggregate 2X, with TP-1 dependent Ring rounds. Network efficiency
defaults to 0.6 and `--comm_startup_us` defaults to 15 us per collective.
Each round charges the maximum of network, L2 and DRAM service time. The API
`run_layer(comm_bandwidth_gbps=...)` uses aggregate Gbit/s, including the TP=4
factor of two; this differs from the CLI's physical-link unit.

TP block totals charge two AG and two RS operators. The cold entrance AG
initializes cache state but is excluded from latency; the charged boundary AG
inherits the final layer norm's warm cache. This does not make the entire compute
pass warm. Thor FP4 charges its entrance quantizer once. Diagnostic next-block
QKV/quantization is excluded, and capacity-eviction writebacks are retained.
TP with degree greater than one bypasses scalar layer caches because they do
not store the L2 state needed across communication boundaries.

For example, from the project root:

```sh
python -m simulate.main --device Thor --model Qwen3_14B --precision fp4 \
  --phase prefill --is_causal --seq_len 1024 --parallelism TP \
  --parallelism_degree 4 --comm_mode serial --comm_bandwidth 256 --icnt_type ucie_std

python -m simulate.main --device Thor --model Qwen3_14B --precision fp4 \
  --phase prefill --is_causal --seq_len 1024 --parallelism TP \
  --parallelism_degree 4 --comm_mode ideal
```

DSE uses TP only in both modes and accepts the same `--comm_mode` option.
The previous `--overlap` option has been removed; `ideal` means zero
communication time, rather than the previous exposed-communication formula.
