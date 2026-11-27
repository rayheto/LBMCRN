# XIAO ESP32-S3

LBMCRN v1.0 was measured on XIAO ESP32-S3 with a 240 MHz CPU and 80 MHz PSRAM. The INT16 package was exported from the released v1.0 weights. Results use the all_psram ESP-DL configuration. GRU lookup tables, gate arrays, input/hidden caches and batch temporaries are explicitly allocated in PSRAM; weights and packing buffers also use PSRAM.

## Runtime

| Measurement | LBMCRN v1.0 |
| --- | ---: |
| Audio hop | 16 ms |
| Independent paced boots | 3 |
| Frames per paced run | 413 |
| Median frame processing time | 9.072 ms |
| Highest p99 across paced runs | 9.109 ms |
| Highest frame time across paced runs | 9.615 ms |
| CPU0 mean, measured RTOS idle runtime | 56.816% |
| CPU1 mean, measured RTOS idle runtime | 0.120% |
| Continuous-run frames | 10,000 |
| Continuous-run p50 / p99 / max | 9.071 / 9.108 / 9.585 ms |
| Frames exceeding 16 ms | 0 |

Each boot had one excluded warmup clip before its paced measurement. CPU utilization is measured over paced runs, excluding I2S. The continuous run is unpaced and checks processing deadlines. Processing time excludes transport and is distinct from end-to-end audio latency.

## Storage and memory

| Measurement | Bytes |
| --- | ---: |
| INT16 deployment package | 133,680 |
| Application binary, excluding model | 1,129,984 |
| Cold-init dynamic internal SRAM increment | 560 |
| Cold-init dynamic PSRAM increment | 279,656 |
| Free internal SRAM after continuous run | 312,975 |
| Free PSRAM after continuous run | 6,251,428 |

Cold-init increments are measured before and after model/DSP initialization. They exclude subsequent audio and timing buffers, and are distinct from peak application use. Free heap is reported separately. Model package size excludes application firmware and partition reservations.

The INT16 package is included at checkpoints/lbmcrn-v1.0-int16.espdl. SHA-256 identities and detailed results are in [DEVICE_RESULTS.json](DEVICE_RESULTS.json).

GRU array allocations use **0 B internal SRAM** and **27,584 B PSRAM**, counting shared tables once. Pointer-region checks cover the tables and lazy first-frame buffers. The 560 B cold-init increment is distinct from GRU arrays and total application memory.

## Export and runtime comparison

The release uses channel-expanded arithmetic PReLU with the original trained slopes. On this board the historical package measured 8.912 ms with LUT-only external memory and 9.058 ms with all GRU arrays external. The selected weights with the corrected arithmetic export measured 9.072 ms with all arrays external, a 0.014 ms difference from the historical package under the same memory profile. [Comparison data](DEVICE_COMPARISON.json).

## Numerical validation

Over 64 consecutive real-audio frames, all 16,512 output mask values exactly matched the saved quantized integer reference. The float-to-ONNX mask and state checks also had zero error on the export test.

These checks establish the tested graph's integer mask behavior. Full PCM quality, human listening and I2S acceptance remain pending.

## Baseline comparison

| Implementation | Audio per call | Processing time per call | CPU0 measured | Model package |
| --- | ---: | ---: | ---: | ---: |
| LBMCRN v1.0, INT16 | 16 ms | 9.072 ms p50 | 56.816% | 133,680 B |
| NSNet2, ESP-SR | 32 ms | 8.503 ms mean | 27.13% | 337,709 B |
| NSNet3, ESP-SR float32 | 16 ms | 11.890 ms mean | 75.00% | 108,740 B |
| GTCRN, experimental ESP-DL INT16 | 16 ms | 37.500 ms p50 | Unmeasured | Unreported |

Baseline rows are historical measurements under different firmware/cache configurations. NSNet2's processing time normalizes to 4.252 ms per 16 ms of audio. The GTCRN result characterizes that implementation, not a minimum attainable latency.

NSNet2's cold-init dynamic heap increments were 8,616 B SRAM and 395,760 B PSRAM. Its dynamic SRAM increment after processing was 10,860 B. NSNet3's cold-init increments were 80 B SRAM and 242,340 B PSRAM, plus 19,048 B of static library data.

Historical baseline records are identified by nsnet_cpu_v1, r2d_sm_nsnet_quality_v1, and gtcrn_deploy_v2 in the archived measurement evidence. Model package formats differ across implementations.
