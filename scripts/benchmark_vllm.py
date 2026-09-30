"""Minimal sequential latency benchmark for the local vLLM server.

This script talks DIRECTLY to the vLLM OpenAI-compatible API:

    POST http://localhost:8000/v1/chat/completions

It deliberately does NOT use FastAPI, the ticket agent, the tool layer,
the database, or the OpenAI Python SDK. It only uses the Python standard
library (urllib for HTTP, time for timing, statistics for the summary), so
it adds no dependencies and it measures the model server alone.

What it does:
    - sends 10 chat-completion requests, one at a time (no concurrency)
    - measures the elapsed time of each request
    - prints the count, min, average, median and max latency

Requirements:
    - vLLM must already be running on http://localhost:8000
      (see DEV.md: "vllm serve Qwen/Qwen3-8B-AWQ --host 0.0.0.0 --port 8000")

Run from the project root, with the virtualenv active:

    .\\.venv\\Scripts\\python.exe scripts\\benchmark_vllm.py

This is a manual script. It is intentionally NOT part of the pytest suite
because it depends on the real Qwen3/vLLM service.
"""

import json
import statistics
import time
import urllib.error
import urllib.request

# Direct vLLM endpoint - no FastAPI in the path.
CHAT_COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"

# Model served by the local vLLM server.
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"

# Sequential requests, one at a time.
NUM_REQUESTS = 10

# A small fixed output length keeps every request a comparable workload.
MAX_TOKENS = 64

REQUEST_TIMEOUT_SECONDS = 120

PROMPT = "Explain what vLLM is in one sentence."


def send_chat_request(prompt: str) -> float:
    """Send one chat-completion request and return its elapsed time in seconds."""
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # The timer covers the whole request: connection, generation, and the body
    # read. That is the latency a real client would experience.
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.read()
    return time.perf_counter() - start


def print_summary(latencies: list) -> None:
    """Print the count, min, average, median and max latency."""
    print()
    print("=" * 60)
    print("benchmark summary (sequential requests to vLLM)")
    print(f"endpoint: {CHAT_COMPLETIONS_URL}")
    print(f"model: {MODEL_NAME}")
    print(f"requests: {len(latencies)}")
    print(f"min latency:     {min(latencies):.3f} s")
    print(f"average latency: {statistics.mean(latencies):.3f} s")
    print(f"median latency:  {statistics.median(latencies):.3f} s")
    print(f"max latency:     {max(latencies):.3f} s")


def main() -> int:
    print("vLLM direct chat-completion benchmark")
    print(f"endpoint: {CHAT_COMPLETIONS_URL}")
    print(f"model: {MODEL_NAME}")
    print(f"sending {NUM_REQUESTS} requests sequentially (one at a time)")
    print()

    latencies = []
    for index in range(1, NUM_REQUESTS + 1):
        try:
            elapsed = send_chat_request(PROMPT)
        except urllib.error.HTTPError as exc:
            print(f"request {index}/{NUM_REQUESTS} failed with HTTP {exc.code}")
            print(exc.read().decode("utf-8", errors="replace"))
            return 1
        except urllib.error.URLError as exc:
            print(f"request {index}/{NUM_REQUESTS} could not reach vLLM: {exc.reason}")
            print("hint: start vLLM on http://localhost:8000 (see DEV.md).")
            return 1

        latencies.append(elapsed)
        print(f"request {index}/{NUM_REQUESTS}: {elapsed:.3f} s")

    print_summary(latencies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
