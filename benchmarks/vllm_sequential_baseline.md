# vLLM Sequential Baseline

## Configuration

- Model: `Qwen/Qwen3-8B-AWQ`
- GPU: RTX 4070 12GB
- Requests: 10 sequential
- Max output tokens: 64
- Endpoint: `http://localhost:8000/v1/chat/completions`

## Results

| Metric | Result |
|---|---:|
| Minimum latency | 2.981 s |
| Average latency | 3.061 s |
| Median latency | 3.064 s |
| Maximum latency | 3.151 s |

## Notes

This benchmark measures end-to-end non-streaming client-observed latency for
sequential requests. Each request waits for the previous request to finish.

The benchmark script is:

`scripts/benchmark_vllm.py`

Date: 2026-09-30