import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.pipeline.ci_gate import CIGate


def find_latest_report(data_dir: str = "eval/data") -> str:
    reports = [f for f in os.listdir(data_dir) if f.endswith(".json") and "eval_report" in f]
    if not reports:
        return None
    reports.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    return os.path.join(data_dir, reports[0])


def main():
    report_path = find_latest_report()
    if not report_path:
        print("❌ No evaluation report found in eval/data/. Run run_eval.py first.")
        sys.exit(1)

    print(f"Loading report: {report_path}")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    gate = CIGate()
    passed, message = gate.check(report)

    if passed:
        print("✅ CI Gate PASSED")
        sys.exit(0)
    else:
        print(f"❌ CI Gate FAILED:\n{message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
