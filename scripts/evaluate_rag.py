"""Evaluate RAG answers against expected facts for each eval case.

Import-time behavior of test_rag_answer.py:
  - Line 25: SentenceTransformer(EMBED_MODEL_NAME) is initialized once.
  - Line 91: _setup_document_embeddings() reads all .md files from
    backend/knowledge_base/, chunks them, and precomputes document
    embeddings into ALL_CHUNKS and CHUNK_EMBEDS.

These side effects run once at import time, before any evaluation logic.
The LLM is NOT called automatically; it is invoked only inside
_generate_answer(), which runs under `if __name__ == "__main__"`.

To run only the local helper tests (no LLM, no retrieval):
    python scripts/evaluate_rag.py --test-only

To run the full evaluation (requires vLLM server at localhost:8000/v1):
    python scripts/evaluate_rag.py
"""

import json
import re
import sys
from pathlib import Path

# Make test_rag_answer importable (same scripts/ directory)
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))

CASES_PATH = scripts_dir / "rag_eval_cases.json"


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def string_match(answer: str, expected_fact: str) -> bool:
    """Preliminary keyword/string match: normalized answer contains normalized fact."""
    return normalize(expected_fact) in normalize(answer)


def source_hit_at_3(expected_source: str | None, filenames: list[str]) -> bool | None:
    """Match an exact filename in the first three results; None means N/A."""
    if expected_source is None:
        return None
    return expected_source in filenames[:3]


# ---------------------------------------------------------------------------
# Explicit phrase matchers for each of the 8 expected facts.
#
# Each matcher config:
#   - fact: the expected fact text (for display only)
#   - required: list of phrases that ALL must be present in the answer
#   - any_of: list of phrases where at least ONE must be present (optional)
#   - description: human-readable explanation of what must be shown
#
# Matching is case-insensitive and whitespace-normalized. A fact is marked
# MATCHED only when every required phrase is present (and any_of, if
# configured, has at least one hit). These are heuristic phrase checks,
# not semantic accuracy measurements.
# ---------------------------------------------------------------------------

FACT_MATCHERS = {
    "return_21_days": [
        {
            "fact": "the return window is 30 days from delivery",
            "required": ["30 days", "delivery"],
            "description": "Must mention the 30-day return window tied to the delivery date",
        },
        {
            "fact": "the laptop must meet the return eligibility conditions",
            "required": ["eligibility conditions"],
            "any_of": ["laptop", "product"],
            "description": "Must reference eligibility conditions applicable to the item being returned",
        },
    ],
    "shipping_no_updates": [
        {
            "fact": "contact customer support for an investigation when there are no tracking updates for 7 or more business days",
            "required": ["customer support", "investigation", "7 or more business days"],
            "any_of": ["no tracking updates", "no updates", "tracking updates"],
            "description": "Must mention contacting support for an investigation after 7+ business days without tracking updates",
        },
    ],
    "shipping_lost_package": [
        {
            "fact": "a reshipment or full refund may be requested after 14 business days from the estimated delivery date",
            "required": ["14 business days", "estimated delivery date", "reshipment", "refund"],
            "description": "Must mention both reshipment AND refund options after 14 business days from the estimated delivery date",
        },
        {
            "fact": "verification is required",
            "required": ["verification"],
            "any_of": ["required", "subject to verification"],
            "description": "Must mention that verification is required",
        },
    ],
    "login_reset_failed": [
        {
            "fact": "escalate to support when password reset still fails after two attempts",
            "required": ["password reset", "two attempts"],
            "any_of": ["escalate", "contact support", "support"],
            "description": "Must mention escalating/contacting support after password reset fails twice",
        },
    ],
    "unsupported_discount": [
        {
            "fact": "the provided policy documents do not establish whether this discount exists",
            "required": ["policy documents", "do not establish", "discount"],
            "description": "Must state that policy documents do not establish whether the discount exists",
        },
        {
            "fact": "do not invent a discount policy",
            "required": ["do not invent", "discount policy"],
            "description": "Must state not to invent a discount policy",
        },
    ],
}


def evaluate_fact(case_id: str, fact_index: int, answer: str) -> tuple[bool, str]:
    """Evaluate whether an answer represents a specific expected fact.

    Uses explicit phrase matchers from FACT_MATCHERS. Returns (matched, reason).
    These are heuristic phrase checks, not semantic accuracy measurements.
    """
    matchers = FACT_MATCHERS.get(case_id, [])
    if fact_index >= len(matchers):
        return False, f"No matcher configured for fact index {fact_index}"

    matcher = matchers[fact_index]
    fact_text = matcher["fact"]
    required = matcher.get("required", [])
    any_of = matcher.get("any_of", [])
    description = matcher.get("description", "")

    norm_answer = normalize(answer)

    # Check all required phrases
    missing_required = [p for p in required if normalize(p) not in norm_answer]
    if missing_required:
        return False, (
            f"FACT: '{fact_text}' — Missing required phrases: {missing_required}. "
            f"({description})"
        )

    # Check any_of phrases (at least one must match)
    if any_of:
        matched_any = [p for p in any_of if normalize(p) in norm_answer]
        if not matched_any:
            return False, (
                f"FACT: '{fact_text}' — Missing one of required alternatives: {any_of}. "
                f"({description})"
            )

    matched_phrases = required + (matched_any if any_of else [])
    return True, (
        f"FACT: '{fact_text}' — All required phrases present: {matched_phrases}. "
        f"({description})"
    )


# ---------------------------------------------------------------------------
# Discount-assertion heuristic (for unsupported_discount manual review).
#
# A correct refusal may repeat "50% discount" while denying that the company
# offers it. We only flag clear affirmative claims. If the signal is mixed or
# absent, we print MANUAL REVIEW REQUIRED rather than asserting wrongness.
# ---------------------------------------------------------------------------

AFFIRMATIVE_DISCOUNT_PATTERNS = [
    r"\byes\b.*\bdiscount\b",
    r"\bdiscount\b.*\byes\b",
    r"we offer.*\bdiscount\b",
    r"the company offers.*\bdiscount\b",
    r"50% discount\s+(?:is|are|available|offer|true)",
    r"\bdiscount of 50%\s+(?:is|are|available|offer|true)",
]

DENIAL_DISCOUNT_PATTERNS = [
    r"do not establish",
    r"not establish",
    r"cannot confirm",
    r"no evidence",
    r"do not invent",
    r"not mentioned",
    r"not specified",
]


def asserts_discount(answer: str) -> str:
    """Determine if the answer affirmatively asserts the company offers the discount.

    Returns one of:
      - "FLAGGED"      : clear affirmative claim, no denial
      - "NOT_FLAGGED"  : clear denial, no affirmative claim (correct refusal)
      - "UNCERTAIN"    : mixed or ambiguous signals — manual review needed
    """
    norm = normalize(answer)

    has_affirmative = any(re.search(p, norm) for p in AFFIRMATIVE_DISCOUNT_PATTERNS)
    has_denial = any(re.search(p, norm) for p in DENIAL_DISCOUNT_PATTERNS)

    if has_affirmative and not has_denial:
        return "FLAGGED"
    if has_denial and not has_affirmative:
        return "NOT_FLAGGED"
    return "UNCERTAIN"


def load_cases(path: Path):
    """Load evaluation cases from JSON. Raise clear error on malformed/missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation cases file not found: {path}\n"
            f"Expected at: {path.resolve()}"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected 'cases' to be a list in {path}")
    for c in cases:
        for field in ("id", "question", "expected_facts", "category", "expected_source"):
            if field not in c:
                raise ValueError(f"Case missing required field '{field}': {c}")
        if not isinstance(c["expected_facts"], list):
            raise ValueError(
                f"Case '{c.get('id', '<unknown>')}' expected_facts must be a list"
            )
        if c["expected_source"] is not None and not isinstance(c["expected_source"], str):
            raise ValueError(
                f"Case '{c['id']}' expected_source must be a filename string or null"
            )
    return cases


# ---------------------------------------------------------------------------
# Local helper tests (no LLM, no retrieval)
# ---------------------------------------------------------------------------


def test_load_cases_expected_source():
    from tempfile import TemporaryDirectory

    case = {"id": "example", "question": "Example?", "expected_facts": [], "category": "test"}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.json"
        for source in ("return_policy.md", None):
            valid_case = {**case, "expected_source": source}
            path.write_text(json.dumps({"cases": [valid_case]}), encoding="utf-8")
            assert load_cases(path) == [valid_case]

        invalid_cases = [case] + [
            {**case, "expected_source": source}
            for source in (123, 1.5, True, [], {})
        ]
        for invalid_case in invalid_cases:
            path.write_text(json.dumps({"cases": [invalid_case]}), encoding="utf-8")
            try:
                load_cases(path)
            except ValueError as exc:
                if "expected_source" not in invalid_case:
                    assert "missing required field 'expected_source'" in str(exc)
                else:
                    assert "Case 'example' expected_source must be a filename string or null" in str(exc)
            else:
                raise AssertionError(f"Expected ValueError for {invalid_case}")
    print("  test_load_cases_expected_source: PASS")


def test_source_hit_at_3():
    filenames = ["return_policy.md", "shipping_policy.md", "troubleshooting_guide.md"]
    for expected_source in filenames:
        assert source_hit_at_3(expected_source, filenames) is True
    assert source_hit_at_3("missing.md", filenames) is False
    assert source_hit_at_3("return_policy.md", []) is False
    assert source_hit_at_3("return_policy.md", ["return_policy.md"] * 3) is True
    assert source_hit_at_3("missing.md", filenames + ["missing.md"]) is False
    assert source_hit_at_3("policy.md", ["return_policy.md"]) is False
    assert source_hit_at_3(None, filenames) is None
    assert source_hit_at_3(None, []) is None
    print("  test_source_hit_at_3: PASS")


def test_normalize():
    assert normalize("  Hello   World  ") == "hello world"
    assert normalize("MiXeD CaSe") == "mixed case"
    assert normalize("") == ""
    assert normalize("\t\n") == ""
    print("  test_normalize: PASS")


def test_string_match():
    assert string_match("The return window is 30 days from delivery.", "30 days") is True
    assert string_match("No mention of returns.", "30 days") is False
    assert string_match("  30   DAYS  from delivery  ", "30 days from delivery") is True
    print("  test_string_match: PASS")


def test_evaluate_fact_return_21_days():
    matched, reason = evaluate_fact("return_21_days", 0,
        "You can return within 30 days of delivery if eligible.")
    assert matched is True, f"Expected match, got: {reason}"

    matched, reason = evaluate_fact("return_21_days", 0,
        "You can return items after delivery.")
    assert matched is False, f"Expected no match, got: {reason}"
    assert "30 days" in reason

    matched, reason = evaluate_fact("return_21_days", 1,
        "The laptop must meet the eligibility conditions.")
    assert matched is True, f"Expected match, got: {reason}"

    matched, reason = evaluate_fact("return_21_days", 1,
        "The laptop must be in good condition.")
    assert matched is False, f"Expected no match, got: {reason}"
    assert "eligibility conditions" in reason

    print("  test_evaluate_fact_return_21_days: PASS")


def test_evaluate_fact_shipping_lost_package():
    matched, reason = evaluate_fact("shipping_lost_package", 0,
        "After 14 business days from the estimated delivery date, "
        "you may request a reshipment or a full refund, subject to verification.")
    assert matched is True, f"Expected match, got: {reason}"

    matched, reason = evaluate_fact("shipping_lost_package", 0,
        "After 14 business days from the estimated delivery date, "
        "you may request a reshipment.")
    assert matched is False, f"Expected no match (missing refund), got: {reason}"
    assert "refund" in reason

    matched, reason = evaluate_fact("shipping_lost_package", 0,
        "After 14 business days from the estimated delivery date, "
        "you may request a full refund.")
    assert matched is False, f"Expected no match (missing reshipment), got: {reason}"
    assert "reshipment" in reason

    matched, reason = evaluate_fact("shipping_lost_package", 1,
        "Verification is required before any reshipment or refund.")
    assert matched is True, f"Expected match, got: {reason}"

    print("  test_evaluate_fact_shipping_lost_package: PASS")


def test_evaluate_fact_unsupported_discount():
    matched, reason = evaluate_fact("unsupported_discount", 0,
        "The policy documents do not establish whether this discount exists.")
    assert matched is True, f"Expected match, got: {reason}"

    matched, reason = evaluate_fact("unsupported_discount", 1,
        "Do not invent a discount policy.")
    assert matched is True, f"Expected match, got: {reason}"

    matched, reason = evaluate_fact("unsupported_discount", 0,
        "The policy documents mention nothing about discounts.")
    assert matched is False, f"Expected no match, got: {reason}"
    assert "do not establish" in reason

    print("  test_evaluate_fact_unsupported_discount: PASS")


def test_asserts_discount():
    result = asserts_discount("Yes, the company offers a 50% discount on all laptops.")
    assert result == "FLAGGED", f"Expected FLAGGED, got: {result}"

    result = asserts_discount(
        "The policy documents do not establish whether a 50% discount exists. "
        "Do not invent a discount policy."
    )
    assert result == "NOT_FLAGGED", f"Expected NOT_FLAGGED, got: {result}"

    result = asserts_discount(
        "The provided policy documents do not establish whether this 50% discount exists."
    )
    assert result == "NOT_FLAGGED", f"Expected NOT_FLAGGED, got: {result}"

    result = asserts_discount(
        "We offer discounts, but I cannot confirm whether a 50% discount exists."
    )
    assert result == "UNCERTAIN", f"Expected UNCERTAIN, got: {result}"

    result = asserts_discount("Thank you for contacting support.")
    assert result == "UNCERTAIN", f"Expected UNCERTAIN, got: {result}"

    print("  test_asserts_discount: PASS")


def run_tests():
    """Run all local helper tests. No LLM, no retrieval."""
    print("Running local helper tests...")
    print("-" * 40)
    test_normalize()
    test_string_match()
    test_load_cases_expected_source()
    test_source_hit_at_3()
    test_evaluate_fact_return_21_days()
    test_evaluate_fact_shipping_lost_package()
    test_evaluate_fact_unsupported_discount()
    test_asserts_discount()
    print("-" * 40)
    print("All local helper tests passed.")


def run_evaluation(cases):
    """Run each case through retrieval + answer generation and print results."""
    # Import here (not at module top) so --test-only works without the model.
    from test_rag_answer import _retrieve_chunks, _generate_answer  # noqa: E402

    total_cases = 0
    total_facts = 0
    matched_facts = 0
    source_hits = 0
    source_cases = 0

    for case in cases:
        case_id = case["id"]
        question = case["question"]
        expected_facts = case["expected_facts"]
        category = case["category"]

        print("=" * 60)
        print(f"Case ID: {case_id}")
        print(f"Category: {category}")
        print(f"Question: {question}")
        print("-" * 60)

        # 1. Retrieve
        retrieved = _retrieve_chunks(question, top_k=3)
        filenames = [chunk["filename"] for chunk, _ in retrieved]
        expected_source = case["expected_source"]
        source_hit = source_hit_at_3(expected_source, filenames)
        if source_hit is None:
            print("Source Hit@3: N/A (no expected source)")
        else:
            source_cases += 1
            source_hits += int(source_hit)
            status = "HIT" if source_hit else "MISS"
            print(f"Source Hit@3: {status} (expected: {expected_source})")

        if not retrieved:
            print("No retrieval data available.")
            continue

        # 2. Print retrieved source filenames
        print(f"Retrieved from: {', '.join(filenames)}")

        # 3. Generate answer (pass pure chunk dicts to preserve prompt format)
        chunk_dicts = [chunk for chunk, _ in retrieved]
        answer = _generate_answer(question, chunk_dicts)
        print(f"\nGenerated Answer:\n{answer}\n")

        # 4. Evaluate each expected fact (heuristic phrase match)
        print("Expected Fact Match (heuristic phrase check — not semantic accuracy):")
        case_fact_matches = 0
        for i, fact in enumerate(expected_facts):
            matched, reason = evaluate_fact(case_id, i, answer)
            status = "MATCH" if matched else "NO MATCH"
            print(f"  [{status}] {fact}")
            print(f"         {reason}")
            total_facts += 1
            if matched:
                matched_facts += 1
                case_fact_matches += 1

        # 5. Unsupported discount manual review flag
        if case_id == "unsupported_discount":
            discount_status = asserts_discount(answer)
            if discount_status == "FLAGGED":
                print(
                    "\n  *** MANUAL REVIEW FLAGGED: Answer may assert the discount exists. ***"
                )
                print(
                    "  A correct refusal may repeat '50% discount'; verify manually."
                )
            elif discount_status == "UNCERTAIN":
                print(
                    "\n  *** MANUAL REVIEW REQUIRED: Discount-claim signal is ambiguous. ***"
                )
                print(
                    "  Verify manually whether the answer affirmatively claims the discount."
                )
            else:
                print(
                    "\n  Discount claim: NOT flagged (answer denies or does not establish the discount)."
                )

        print(f"\nCase facts matched: {case_fact_matches}/{len(expected_facts)}")
        total_cases += 1

    # Summary
    print("\n" + "=" * 60)
    print("PRELIMINARY HEURISTIC SUMMARY")
    print("=" * 60)
    print(f"Cases run successfully: {total_cases}/{len(cases)}")
    print(f"Expected facts matched: {matched_facts}/{total_facts}")
    if source_cases:
        print(f"Source Hit@3: {source_hits}/{source_cases} ({source_hits / source_cases:.1%})")
    else:
        print("Source Hit@3: N/A (no supported cases)")
    print("\nNOTE: These are preliminary keyword/phrase-match metrics only.")
    print("      They do not prove semantic correctness.")


def main():
    # Support --test-only to run helper tests without LLM
    if "--test-only" in sys.argv:
        run_tests()
        return

    try:
        cases = load_cases(CASES_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    run_evaluation(cases)


if __name__ == "__main__":
    main()
