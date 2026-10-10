"""
CI Evaluation Gate - runs golden dataset and fails if thresholds not met.
Supports baseline comparison and PR comment generation.
"""
import json
import sys
import os
import requests

API_URL      = os.getenv("EVAL_API_URL", "http://localhost:8000/api/v1/eval/golden-run")
API_KEY      = os.getenv("EVAL_API_KEY", "")
REPORT_PATH  = "ci/eval_report.json"
BASELINE_PATH= "ci/eval_baseline.json"
MIN_PASS_RATE= 0.80


def post_pr_comment(report: dict) -> None:
    """Post summary comment to GitHub PR if token is set."""
    token  = os.getenv("GITHUB_TOKEN", "")
    repo   = os.getenv("GITHUB_REPOSITORY", "")
    pr_num = os.getenv("PR_NUMBER", "")

    if not all([token, repo, pr_num]):
        return

    body = f"""## LLM Evaluation Quality Gate Report

| Metric | Value |
|--------|-------|
| Total Cases | {report['total']} |
| Passed | {report['passed']} |
| Failed | {report['failed']} |
| Pass Rate | {report['pass_rate']:.1%} |
| Gate Status | {'✅ PASSED' if report['ci_passed'] else '❌ FAILED'} |
"""
    try:
        requests.post(
            f"https://api.github.com/repos/{repo}/issues/{pr_num}/comments",
            json={"body": body},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except Exception as e:
        print(f"[warn] Could not post PR comment: {e}")


def compare_baseline(current: dict, baseline: dict) -> None:
    """Print regression summary vs baseline."""
    print("\nBaseline Comparison:")
    for key in ["pass_rate"]:
        curr_val = current.get(key, 0)
        base_val = baseline.get(key, 0)
        delta    = curr_val - base_val
        symbol   = "▲" if delta >= 0 else "▼"
        print(f"  pass_rate: {curr_val:.1%}  (baseline {base_val:.1%})  {symbol}{abs(delta):.1%}")


def main():
    print("=" * 60)
    print("LLM Evaluation Quality Gate - Golden Dataset Run")
    print("=" * 60)

    headers = {"X-API-Key": API_KEY} if API_KEY else {}

    try:
        resp = requests.post(API_URL, json={}, headers=headers, timeout=600)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[FATAL] Could not reach evaluation API: {e}")
        sys.exit(1)

    # Write report artifact
    os.makedirs("ci", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nReport saved to {REPORT_PATH}")

    # Baseline comparison
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        compare_baseline(data, baseline)
    else:
        # First run - save as baseline
        with open(BASELINE_PATH, "w") as f:
            json.dump({"pass_rate": data["pass_rate"]}, f, indent=2)
        print("\nFirst run - saved as baseline.")

    print(f"\nTotal cases  : {data['total']}")
    print(f"Passed       : {data['passed']}")
    print(f"Failed       : {data['failed']}")
    print(f"Pass rate    : {data['pass_rate']:.1%}")
    print(f"Threshold    : {MIN_PASS_RATE:.1%}")

    post_pr_comment(data)

    if data["ci_passed"]:
        print("\n✅ QUALITY GATE PASSED - merge is allowed.")
        sys.exit(0)
    else:
        print(f"\n❌ QUALITY GATE FAILED - pass rate {data['pass_rate']:.1%} below {MIN_PASS_RATE:.1%}.")
        sys.exit(1)


if __name__ == "__main__":
    main()