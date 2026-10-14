"""LIVE evaluation of English / Swahili / Luganda handling (uses real API keys and costs money).

Runs each claim in tests/data/multilingual_claims.json through the real pipeline and reports:
  - whether the claim was extracted without being translated or altered,
  - the verdict vs. the expected verdict,
  - how many evidence items were found.
Never treat these results as language parity unless they hold on a larger, native-speaker-reviewed set.

Usage (from the project root, virtual environment active, .env configured):
    python script/eval_multilingual.py               # all cases
    python script/eval_multilingual.py --only sw     # one language
Estimated cost: roughly 10 claims x (3-4 searches + ~10 AI calls). With gpt-4.1-mini, typically well under US$0.50.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factcheck import FactCheck  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data", "multilingual_claims.json")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["en", "sw", "lg"], default=None)
    parser.add_argument("--model", default=os.environ.get("TRUTHLENS_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--client", default=None)
    parser.add_argument("--out", default="multilingual_eval_results.json")
    args = parser.parse_args()

    with open(DATA, encoding="utf-8") as f:
        cases = [c for c in json.load(f)["cases"] if not args.only or c["language"] == args.only]
    fc = FactCheck(default_model=args.model, client=args.client)

    rows = []
    for case in cases:
        try:
            res = fc.check_text(case["claim"], country=case["country"], language=case["language"], output_language=case["language"])
            claims = res["claim_detail"]
            extracted = [c["claim"] for c in claims]
            verdict = claims[0]["verdict"] if claims else "NO_CLAIM"
            preserved = any(case["claim"].rstrip(".").lower() in c.lower() or c.rstrip(".").lower() in case["claim"].lower() for c in extracted)
            n_evidence = sum(len(c["evidences"]) for c in claims)
            rows.append({"id": case["id"], "language": case["language"], "expected": case["expected_verdict"], "verdict": verdict,
                         "correct": verdict == case["expected_verdict"], "claim_preserved": preserved,
                         "extracted": extracted, "evidence": n_evidence,
                         "explanation": claims[0]["explanation"] if claims else ""})
        except Exception as e:
            rows.append({"id": case["id"], "language": case["language"], "expected": case["expected_verdict"],
                         "verdict": f"ERROR: {e}", "correct": False, "claim_preserved": False, "extracted": [], "evidence": 0})
        r = rows[-1]
        print(f"{r['id']:6} expected={r['expected']:14} got={r['verdict']:22} preserved={r['claim_preserved']} evidence={r['evidence']}")

    for lang in sorted({r["language"] for r in rows}):
        sub = [r for r in rows if r["language"] == lang]
        print(f"[{lang}] verdict accuracy {sum(r['correct'] for r in sub)}/{len(sub)}, "
              f"claims preserved {sum(r['claim_preserved'] for r in sub)}/{len(sub)}")
    print("INSUFFICIENT_EVIDENCE on a true/false claim is a cautious miss, not a wrong verdict - inspect each report.")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Details written to {args.out}")


if __name__ == "__main__":
    main()
