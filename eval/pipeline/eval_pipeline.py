from typing import List, Dict, Any, Optional
from eval.pipeline.rag_evaluator import RAGEvaluator
from eval.pipeline.generation_evaluator import GenerationEvaluator
from eval.pipeline.control_evaluator import ControlEvaluator
from eval.pipeline.hallucination_evaluator import HallucinationEvaluator
from eval.pipeline.dialog_evaluator import DialogCohesionEvaluator
from eval.pipeline.ci_gate import CIGate
from eval.pipeline.report_generator import ReportGenerator


class EvalPipeline:
    def __init__(
        self,
        output_dir: str = "eval/data",
        ci_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.output_dir = output_dir
        self.rag_evaluator = RAGEvaluator()
        self.gen_evaluator = GenerationEvaluator()
        self.ctrl_evaluator = ControlEvaluator()
        self.hall_evaluator = HallucinationEvaluator()
        self.dialog_evaluator = DialogCohesionEvaluator()
        ci_kwargs = ci_thresholds or {}
        self.ci_gate = CIGate(**ci_kwargs)
        self.report_generator = ReportGenerator(output_dir=output_dir)

    def run(
        self,
        gt_samples: List[Dict],
        model_outputs: List[Dict],
        report_name: str = "eval_report",
        mode: str = "dataset",
        dialog_sessions: Optional[List[List[Dict]]] = None,
    ) -> Dict[str, Any]:
        rag_results = self.rag_evaluator.evaluate(gt_samples, model_outputs)
        predictions = [o.get("generated_answer", "") for o in model_outputs]
        references = [g.get("answer", "") for g in gt_samples[:len(predictions)]]
        gen_results = self.gen_evaluator.evaluate(predictions, references)
        hall_results = self.hall_evaluator.evaluate(gt_samples, model_outputs)
        ctrl_results = self.ctrl_evaluator.evaluate(gt_samples, model_outputs)
        all_results = {**rag_results, **gen_results, **hall_results, **ctrl_results}
        all_results["rag_recall_at_5"] = rag_results.get("context_recall", 0.0)
        if dialog_sessions is not None:
            dialog_results = self.dialog_evaluator.evaluate(dialog_sessions)
            all_results.update(dialog_results)
        report = self.report_generator.generate(all_results, report_name=report_name)
        passed, gate_message = self.ci_gate.check(all_results)
        report["ci_gate_passed"] = passed
        report["ci_gate_message"] = gate_message
        return report
