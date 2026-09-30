"""Manual REAL-MODEL evaluation of the Qwen3 tool-calling agent.

This calls the EXISTING running application endpoint:

    POST http://localhost:8080/tickets/{ticket_id}/ask

so it evaluates the complete real path:

    HTTP -> FastAPI -> tool_agent_service -> tools -> repositories
         -> Supabase PostgreSQL -> vLLM/Qwen3 (port 8000)

It does NOT call vLLM directly, and it makes no database changes.

Grading per case (final-answer quality is always ungraded):
    - tool sequence: the tool names in verified_facts, in order
    - verified results: the exact verified_facts (tool + result) returned
      by the API
A case is PASS only when BOTH checks match.

REAL-MODEL REQUIREMENTS (both services must be running):
    - FastAPI backend on http://localhost:8080
    - vLLM serving Qwen3 (Qwen3-8B-AWQ) on http://localhost:8000

Run from the project root, with the virtualenv active:

    .\\.venv\\Scripts\\python.exe scripts\\evaluate_qwen_agent.py

This is a manual script. It is intentionally NOT part of the pytest suite
because it depends on the real Qwen3/vLLM service.
"""

import json
import socket
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8080"
ASK_URL = BASE_URL + "/tickets/{ticket_id}/ask"
REQUEST_TIMEOUT_SECONDS = 120

# Real evaluation cases.
# ``expected_tools`` grades the tool sequence; ``expected_facts`` grades the
# exact verified_facts the API returns (a list of {"tool", "result"} dicts).
CASES = [
    {
        "name": "case1_login_no_tool",
        "ticket_id": 1,
        "question": "I cannot log into my account. What can I try?",
        "expected_tools": [],
        "expected_facts": [],
    },
    {
        "name": "case2_order_status",
        "ticket_id": 3,
        "question": "What is the current status of my order ORD-12345?",
        "expected_tools": ["get_order_status"],
        "expected_facts": [
            {
                "tool": "get_order_status",
                "result": {
                    "product": "Laptop",
                    "status": "shipped",
                    "payment_id": "PAY-88888",
                },
            },
        ],
    },
    {
        "name": "case3_order_then_payment",
        "ticket_id": 3,
        "question": (
            "Was the payment for ORD-12345 successful, and what is the "
            "current order status?"
        ),
        "expected_tools": ["get_order_status", "get_payment_status"],
        "expected_facts": [
            {
                "tool": "get_order_status",
                "result": {
                    "product": "Laptop",
                    "status": "shipped",
                    "payment_id": "PAY-88888",
                },
            },
            {
                "tool": "get_payment_status",
                "result": {
                    "status": "captured",
                    "amount": 1299.0,
                    "currency": "MYR",
                },
            },
        ],
    },
    {
        "name": "case4_missing_identifier",
        "ticket_id": 2,
        "question": "Why was I charged twice this month?",
        "expected_tools": [],
        "expected_facts": [],
    },
    {
        "name": "case5_order_not_found",
        "ticket_id": 3,
        "question": "Please check ORD-99999.",
        "expected_tools": ["get_order_status"],
        "expected_facts": [
            {
                "tool": "get_order_status",
                "result": {"error": "Order not found"},
            },
        ],
    },
]


def ask(ticket_id, question):
    """POST the question to the running app. Returns (status, json_payload)."""
    url = ASK_URL.format(ticket_id=ticket_id)
    body = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _report_timeout(case, reason=None):
    """Report a real-model request timeout without grading it as a FAIL."""
    print("result: TIMEOUT")
    print(f"case: {case['name']}")
    print(
        "the real-model request exceeded the evaluation timeout "
        f"({REQUEST_TIMEOUT_SECONDS}s)."
    )
    if reason is not None:
        print(f"timeout detail: {reason}")
    print("the FastAPI/vLLM request may still be processing.")
    print("suggested next checks:")
    print("  - FastAPI terminal: is the /ask request still running? any traceback?")
    print("  - vLLM terminal: is the Qwen3 server still generating or stuck?")


def run_case(case):
    """Run one case and grade it.

    Returns a dict with "overall", "tools", and "results" keys, each being
    "PASS", "FAIL", or "TIMEOUT". "overall" is "PASS" only when both the tool
    sequence and the verified results match. Timeouts are reported separately
    and are never counted as a FAIL.
    """
    print("=" * 72)
    print(f"CASE: {case['name']}")
    print(f"ticket_id: {case['ticket_id']}")
    print(f"question: {case['question']}")
    print(f"expected tools: {case['expected_tools']}")
    print("expected verified_facts:")
    print(json.dumps(case["expected_facts"], indent=2))
    print("-" * 72)

    try:
        status, payload = ask(case["ticket_id"], case["question"])
    except (socket.timeout, TimeoutError) as exc:
        _report_timeout(case, reason=exc)
        return {"overall": "TIMEOUT", "tools": "TIMEOUT", "results": "TIMEOUT"}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace")
        print(f"HTTP status: {exc.code}")
        print("FAIL: the endpoint returned a non-2xx response.")
        print(f"response body: {error_body}")
        if exc.code == 503:
            print("hint: 503 usually means vLLM/Qwen3 on port 8000 is unavailable.")
        if exc.code == 504:
            print("hint: 504 suggests an upstream timeout (FastAPI -> vLLM).")
        print(
            "no verified_facts available: tool-sequence and verified-result "
            "are graded FAIL."
        )
        return {"overall": "FAIL", "tools": "FAIL", "results": "FAIL"}
    except urllib.error.URLError as exc:
        # urllib wraps many socket errors, including timeouts, in URLError.
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            _report_timeout(case, reason=exc.reason)
            return {
                "overall": "TIMEOUT",
                "tools": "TIMEOUT",
                "results": "TIMEOUT",
            }
        print("HTTP status: (no response)")
        print(f"FAIL: could not reach the FastAPI backend at {BASE_URL}.")
        print(f"reason: {exc.reason}")
        print(
            "hint: start the backend from the project root: "
            "uvicorn backend.main:app --reload --port 8080"
        )
        print(
            "no verified_facts available: tool-sequence and verified-result "
            "are graded FAIL."
        )
        return {"overall": "FAIL", "tools": "FAIL", "results": "FAIL"}

    print(f"HTTP status: {status}")
    verified_facts = payload.get("verified_facts", [])
    # Tool sequence is taken from verified_facts (not from log text).
    actual_tools = [fact.get("tool") for fact in verified_facts]

    print("actual verified_facts:")
    print(json.dumps(verified_facts, indent=2))
    print(f"actual tool sequence: {actual_tools}")
    print(f"final answer: {payload.get('answer')}")

    tools_match = actual_tools == case["expected_tools"]
    # Verified results are compared against the exact verified_facts the API
    # returned (never inferred from the final answer text).
    results_match = verified_facts == case["expected_facts"]

    tools_outcome = "PASS" if tools_match else "FAIL"
    results_outcome = "PASS" if results_match else "FAIL"

    print(f"tool-sequence match: {tools_outcome}")
    print(f"verified-result match: {results_outcome}")
    print("note: final-answer quality is NOT graded here.")
    print("      grading covers only the tool sequence and verified results.")

    # A case passes only when BOTH checks match.
    overall = "PASS" if tools_match and results_match else "FAIL"
    print(f"case result: {overall}")
    return {
        "overall": overall,
        "tools": tools_outcome,
        "results": results_outcome,
    }


def main() -> int:
    print("REAL-MODEL evaluation of the Qwen3 tool-calling agent")
    print(f"target endpoint: POST {BASE_URL}/tickets/{{ticket_id}}/ask")
    print("requires FastAPI on port 8080 and vLLM/Qwen3 on port 8000")
    print("this script does not modify the database.")
    print()

    overall_counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    tools_counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    results_counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    for case in CASES:
        outcome = run_case(case)
        overall_counts[outcome["overall"]] += 1
        tools_counts[outcome["tools"]] += 1
        results_counts[outcome["results"]] += 1
        print()

    total = len(CASES)
    print("=" * 72)
    print(f"tool-sequence PASS: {tools_counts['PASS']}/{total}")
    print(f"tool-sequence FAIL: {tools_counts['FAIL']}/{total}")
    print(f"verified-result PASS: {results_counts['PASS']}/{total}")
    print(f"verified-result FAIL: {results_counts['FAIL']}/{total}")
    print(f"TIMEOUT (not graded): {overall_counts['TIMEOUT']}/{total}")
    print(
        f"overall case PASS (tools AND results): "
        f"{overall_counts['PASS']}/{total}"
    )
    print("reminder: final-answer quality is NOT graded by this script.")
    print("grading covers only the tool sequence and verified results.")
    return 0 if overall_counts["PASS"] == total else 1


if __name__ == "__main__":
    raise SystemExit(main())