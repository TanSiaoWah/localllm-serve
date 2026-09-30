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

Two requests completed in approximately **3.07 seconds total**, which is close to the single-request sequential baseline of **3.06 seconds**.

This suggests that the local vLLM server handled these two concurrent requests efficiently rather than simply processing them one after another.

This result alone does not establish the exact internal scheduling mechanism used by vLLM.


## 3. Four-Request Concurrency

### Results

| Metric | Result |
|---|---:|
| Total elapsed time | 3.369 s |
| Successful requests | 4 / 4 |

Individual request latencies:

| Request | Latency |
|---|---:|
| 1 | 3.365 s |
| 2 | 3.346 s |
| 3 | 3.364 s |
| 4 | 3.364 s |

### Interpretation

Four concurrent requests completed in approximately **3.37 seconds total**,
which is only slightly slower than the two-request concurrent result of
**3.07 seconds**. This indicates that the local vLLM server handled the
additional concurrent load efficiently rather than processing all requests
strictly one after another.


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
