from typing import List, Dict, Any, Optional
import json
import os
import random


class ReportGenerator:
    def __init__(self, output_dir: str = "eval/data", bootstrap_n: int = 1000):
        self.output_dir = output_dir
        self.bootstrap_n = bootstrap_n
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, results: Dict[str, float], report_name: str = "eval_report") -> Dict[str, Any]:
        json_path = os.path.join(self.output_dir, f"{report_name}.json")
        md_path = os.path.join(self.output_dir, f"{report_name}.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        md_content = self._generate_markdown(results, report_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return results

    def generate_with_ci(
        self,
        results: Dict[str, float],
        sample_values: Dict[str, List[float]],
        report_name: str = "eval_ci_report",
    ) -> Dict[str, Any]:
        report = dict(results)
        for metric, values in sample_values.items():
            if not values:
                continue
            bootstrapped = []
            for _ in range(min(self.bootstrap_n, 100)):
                sample = [random.choice(values) for _ in range(len(values))]
                bootstrapped.append(sum(sample) / len(sample))
            bootstrapped.sort()
            lower_idx = max(0, int(len(bootstrapped) * 0.025))
            upper_idx = min(len(bootstrapped) - 1, int(len(bootstrapped) * 0.975))
            report[f"{metric}_ci_lower"] = bootstrapped[lower_idx]
            report[f"{metric}_ci_upper"] = bootstrapped[upper_idx]
        report["ci_lower"] = True
        report["ci_upper"] = True
        json_path = os.path.join(self.output_dir, f"{report_name}.json")
        md_path = os.path.join(self.output_dir, f"{report_name}.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        md_content = self._generate_markdown(report, report_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return report

    def _generate_markdown(self, results: Dict[str, Any], title: str) -> str:
        lines = [f"# Evaluation Report: {title}\n"]
        lines.append("## Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in results.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
        return "\n".join(lines)
