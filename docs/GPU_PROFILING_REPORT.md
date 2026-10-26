# GPU Profiling and Bottleneck Analysis

## 1. Overview

This report documents GPU performance profiling of a vLLM-based large language model inference service.

**Objective:** Understand GPU execution characteristics under prefill-heavy and decode-heavy workloads, identify expensive CUDA kernel families, and investigate hardware-level performance limitations.

### Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB |
| Operating System | Ubuntu 24.04 on WSL2 |
| Model | Qwen2.5-1.5B-Instruct |
| Serving Engine | vLLM 0.32.0 |
| Model Precision | FP16 |
| Maximum Model Length | 2048 tokens |
| Prefix Caching | Disabled |
| GPU Trace Profiler | PyTorch Profiler |
| Hardware Profiler | NVIDIA Nsight Compute 2026.3.1 |

## 2. Workload Design

We collected GPU execution traces for two representative inference workloads.

| Configuration | Prefill-heavy | Decode-heavy |
|---|---:|---:|
| Input Tokens / Request | 1024 | 128 |
| Output Tokens / Request | 64 | 512 |
| Concurrent Requests | 8 | 8 |
| Total Requests | 8 | 8 |
| Total Input Tokens | 8192 | 1024 |
| Total Output Tokens | 512 | 4096 |

Both workloads used the same model and serving configuration.

Profiling started after a warmup request and stopped after all eight measurement requests completed.

The experiments were designed to expose contrasting execution patterns, rather than establish production-level latency or throughput.

## 3. GPU Trace Analysis

### 3.1 Recorded GPU Activity

| Metric | Prefill-heavy | Decode-heavy |
|---|---:|---:|
| GPU Activity Span | 779.27 ms | 3347.10 ms |
| Activity Coverage | 96.59% | 96.54% |
| GPU Activity Events | 27,532 | 208,669 |
| Activity Union / 1K Output Tokens | 1470.05 ms | 788.86 ms |

Both workloads exhibited approximately 96.5% recorded GPU activity coverage.

This indicates that the captured execution windows contained relatively few long gaps without GPU activity.

However, activity coverage is not equivalent to SM utilization, achieved occupancy, or compute throughput.

The normalized activity duration includes both prefill and decoding operations; therefore, it should not be interpreted as an isolated per-token decode cost.

### 3.2 Kernel Family Breakdown

| Kernel Family | Prefill-heavy Duration | Decode-heavy Duration |
|---|---:|---:|
| GEMM / CUTLASS | 633.77 ms | 2840.14 ms |
| FlashAttention | 67.32 ms | 202.41 ms |
| Triton | 36.02 ms | 87.92 ms |
| KV Cache | 3.75 ms | 23.08 ms |
| Memory Transfer | 1.09 ms | 6.36 ms |

GEMM / CUTLASS accounted for the largest proportion of recorded kernel execution time in both workloads.

The decode-heavy trace contained substantially more GPU events, consistent with its greater number of autoregressive generation steps.

Kernel classification was performed using name-based heuristics. Duration totals may include activities that overlap in time and must not be interpreted as wall-clock execution time.

## 4. Hardware-Level Profiling

We used NVIDIA Nsight Compute to investigate hardware utilization and warp scheduling behavior.

Hardware-level profiling was performed on standalone PyTorch GEMM operations, not directly on the vLLM service.

### 4.1 FP32 vs. FP16 GEMM

Both experiments multiplied 2048 × 2048 matrices.

| Metric | FP32 SIMT GEMM | FP16 Tensor Core GEMM |
|---|---:|---:|
| Kernel Duration | 582.11 μs | 213.73 μs |
| SM Throughput | 77.94% | 46.11% |
| DRAM Throughput | 9.23% | 12.03% |
| Achieved Occupancy | 16.63% | 16.66% |
| Registers / Thread | 202 | 218 |
| Dynamic Shared Memory / Block | 49.15 KB | 73.73 KB |

The measured FP16 GEMM kernel was approximately 2.72 times faster than the measured FP32 GEMM kernel.

However, these results reflect different kernel implementations and precision formats. They do not establish a universal FP16 speedup factor.

Despite lower measured SM throughput, the FP16 Tensor Core kernel completed sooner.

### 4.2 FP16 Warp Scheduling

Nsight Compute reported the following for the FP16 GEMM kernel:

| Metric | Value |
|---|---:|
| Active Warps / Scheduler | 2.00 |
| Eligible Warps / Scheduler | 0.10 |
| Scheduler Cycles with No Eligible Warp | 93.21% |
| Issue Slots Busy | 6.53% |
| Warp Stall Contribution from Math Pipeline Throttle | 82.4% |

These observations suggest limited latency-hiding opportunities in the measured kernel.

High register and shared-memory usage restrict the number of resident warps. Math pipeline contention or dependency-related waiting can further reduce the number of warps eligible to issue instructions.

The observed low occupancy and warp stalls are useful diagnostic evidence, but they do not establish a single definitive root cause.

The profiler's estimated potential speedups should not be treated as achievable benchmark improvements without an optimization experiment.

## 5. Key Findings

**Finding 1 — Matrix operations dominate recorded kernel execution time.**

GEMM-related kernel families account for most of the accumulated GPU execution time in both profiled workloads.

**Finding 2 — Workload shape materially changes GPU execution behavior.**

Long-input, short-output and short-input, long-output workloads produce different kernel counts, execution durations, and amortized GPU activity costs.

**Finding 3 — High activity coverage does not imply full hardware utilization.**

The GPU timeline shows nearly continuous recorded activity, but hardware metrics such as SM throughput and occupancy must be measured separately.

**Finding 4 — A standalone FP16 GEMM exhibits limited warp-level parallelism.**

The measured Tensor Core GEMM has low achieved occupancy and a high percentage of scheduler cycles without eligible warps.

These findings motivate more targeted investigation of execution shapes, batching, and kernel efficiency.

## 6. Limitations

- PyTorch Profiler introduces instrumentation overhead.
- GPU Activity Coverage is not a hardware utilization metric.
- Kernel-family classification is heuristic.
- Kernel duration sums can include overlapping execution.
- The normalized GPU activity metrics include prefill costs.
- Nsight Compute may replay kernels and alter observed execution conditions.
- Hardware counters were collected from standalone PyTorch GEMM, not actual vLLM inference kernels.
- No controlled comparison of different vLLM scheduling policies was conducted.
- The analysis does not establish that the complete serving system is compute-bound or memory-bound.

## 7. Conclusion

We developed a reproducible GPU profiling workflow combining PyTorch execution traces, offline kernel analysis, and NVIDIA Nsight Compute hardware counters.

The work identified GEMM-related execution as a major contributor to recorded GPU time, characterized differences between prefill-heavy and decode-heavy workloads, and demonstrated hardware-level investigation of occupancy and warp scheduling.

The results provide a foundation for future targeted inference optimizations, including batching strategies, scheduling analysis, and kernel-level investigations.

**Status:** GPU profiling and baseline hardware diagnosis completed.