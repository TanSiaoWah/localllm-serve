# vLLM Benchmark Results

## Configuration

* Model: `Qwen/Qwen3-8B-AWQ`
* GPU: RTX 4070 12GB
* Endpoint: `http://localhost:8000/v1/chat/completions`
* Prompt: `Explain what vLLM is in one sentence.`
* Maximum output tokens: `64`
* Benchmark script: `scripts/benchmark_vllm.py`
* FastAPI: not involved
* Database: not involved
* Supabase: not involved

The benchmark measures direct client-observed latency to the vLLM OpenAI-compatible API.

---

## 1. Sequential Baseline

### Workload

* 10 requests
* Requests are sent one at a time
* Each request waits for the previous request to finish

### Results

| Metric          |  Result |
| --------------- | ------: |
| Minimum latency | 2.923 s |
| Average latency | 3.063 s |
| Median latency  | 3.066 s |
| Maximum latency | 3.239 s |

### Interpretation

For this workload, a single request takes approximately **3.06 seconds on average**.

---

## 2. Two-Request Concurrency

### Workload

* Concurrency level: 2
* Two requests are sent at the same time
* Same model, prompt, and `max_tokens` as the sequential benchmark

### Results

| Metric              |  Result |
| ------------------- | ------: |
| Total elapsed time  | 3.071 s |
| Successful requests |   2 / 2 |

Individual request latencies:

| Request | Latency |
| ------- | ------: |
| 1       | 3.065 s |
| 2       | 3.038 s |

### Interpretation

Two concurrent requests completed in approximately **3.07 seconds total**, which is close to the single-request sequential baseline of **3.06 seconds**.

This indicates that the local vLLM server handled two concurrent requests efficiently rather than simply processing them strictly one after another.

---

## 3. Four-Request Concurrency

### Workload

* Concurrency level: 4
* Four requests are sent at the same time
* Same model, prompt, and `max_tokens` as the sequential benchmark

### Results

| Metric              |  Result |
| ------------------- | ------: |
| Total elapsed time  | 3.369 s |
| Successful requests |   4 / 4 |

Individual request latencies:

| Request | Latency |
| ------- | ------: |
| 1       | 3.365 s |
| 2       | 3.346 s |
| 3       | 3.364 s |
| 4       | 3.364 s |

### Interpretation

Four concurrent requests completed in approximately **3.37 seconds total**, which is only slightly slower than the two-request result of **3.07 seconds**.

This indicates that the local vLLM server continued to handle the additional concurrent load efficiently at concurrency 4.

---

## 4. Eight-Request Concurrency

### Workload

* Concurrency level: 8
* Eight requests are sent at the same time
* Same model, prompt, and `max_tokens` as the sequential benchmark

### Results

| Metric              |   Result |
| ------------------- | -------: |
| Total elapsed time  | 10.597 s |
| Successful requests |    8 / 8 |

Individual request latencies:

| Request |  Latency |
| ------- | -------: |
| 1       | 10.592 s |
| 2       | 10.592 s |
| 3       | 10.591 s |
| 4       | 10.573 s |
| 5       | 10.548 s |
| 6       | 10.590 s |
| 7       | 10.590 s |
| 8       | 10.571 s |

### Interpretation

At concurrency 8, total elapsed time increased substantially to approximately **10.60 seconds**, compared with **3.37 seconds** at concurrency 4.

All eight requests still completed successfully, but the larger latency indicates that the inference server no longer maintained the near-constant wall-clock behavior observed at lower concurrency.

This benchmark does not measure GPU utilization, so it does not by itself establish the exact hardware or scheduler bottleneck causing the increase.

---

## 5. Concurrency Summary

| Workload             | Total elapsed time | Successful requests |
| -------------------- | -----------------: | ------------------: |
| 1 sequential request |             ~3.1 s |               1 / 1 |
| 2 concurrent         |            3.071 s |               2 / 2 |
| 4 concurrent         |            3.369 s |               4 / 4 |
| 8 concurrent         |           10.597 s |               8 / 8 |

### Overall Observation

The benchmark shows three useful stages:

1. **Low concurrency (2):** latency remains close to the single-request baseline.
2. **Moderate concurrency (4):** latency increases only slightly.
3. **Higher concurrency (8):** latency increases substantially.

This suggests that the local Qwen3-8B-AWQ deployment handles low-to-moderate concurrency efficiently, while higher concurrency introduces a significant latency penalty.

The benchmark stops at concurrency 8 because the current measurements already demonstrate the change in behavior without adding unnecessary load.

---

## Notes

`max_tokens=64` is a maximum output-token limit, not a fixed output length. It bounds the amount of text the model can generate so that benchmark requests have a more comparable workload.

The latency timer covers the complete non-streaming request from the Python client perspective:

```text
HTTP request
+ prompt processing
+ model generation
+ HTTP response transfer
+ response body reading
```

The benchmark uses Python's standard library only and does not add any project dependencies.

### Benchmark Date

`2026-09-30`
