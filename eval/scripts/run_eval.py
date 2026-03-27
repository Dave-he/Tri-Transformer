import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.pipeline.eval_pipeline import EvalPipeline


def load_gt_from_jsonl(gt_file: str) -> list:
    samples = []
    with open(gt_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline")
    parser.add_argument("--mode", choices=["single", "batch", "dataset"], default="dataset")
    parser.add_argument("--gt_file", required=True, help="Path to Ground Truth JSONL file")
    parser.add_argument("--output_dir", default="eval/data", help="Output directory for reports")
    parser.add_argument("--report_name", default="eval_report")
    args = parser.parse_args()

    print(f"Loading Ground Truth from {args.gt_file}")
    gt_samples = load_gt_from_jsonl(args.gt_file)
    print(f"Loaded {len(gt_samples)} GT samples")

    model_outputs = []
    for gt in gt_samples:
        model_outputs.append({
            "query": gt.get("query", ""),
            "generated_answer": gt.get("answer", ""),
            "retrieved_docs": gt.get("source_docs", []),
        })

    pipeline = EvalPipeline(output_dir=args.output_dir)
    report = pipeline.run(gt_samples, model_outputs, report_name=args.report_name, mode=args.mode)

    print("\n=== Evaluation Results ===")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif k not in ["ci_gate_message"]:
            print(f"  {k}: {v}")

    if not report.get("ci_gate_passed", True):
        print(f"\n❌ CI Gate FAILED:\n{report.get('ci_gate_message', '')}")
        sys.exit(1)
    else:
        print("\n✅ CI Gate PASSED")


if __name__ == "__main__":
    main()
