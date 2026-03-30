def __getattr__(name):
    if name == "EvalPipeline":
        from eval.pipeline.eval_pipeline import EvalPipeline
        return EvalPipeline
    if name == "RAGEvaluator":
        from eval.pipeline.rag_evaluator import RAGEvaluator
        return RAGEvaluator
    if name == "GenerationEvaluator":
        from eval.pipeline.generation_evaluator import GenerationEvaluator
        return GenerationEvaluator
    if name == "HallucinationEvaluator":
        from eval.pipeline.hallucination_evaluator import HallucinationEvaluator
        return HallucinationEvaluator
    if name == "ControlEvaluator":
        from eval.pipeline.control_evaluator import ControlEvaluator
        return ControlEvaluator
    if name == "DialogCohesionEvaluator":
        from eval.pipeline.dialog_evaluator import DialogCohesionEvaluator
        return DialogCohesionEvaluator
    if name == "CIGate":
        from eval.pipeline.ci_gate import CIGate
        return CIGate
    if name == "ReportGenerator":
        from eval.pipeline.report_generator import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module 'eval.pipeline' has no attribute {name!r}")


__all__ = [
    "EvalPipeline",
    "RAGEvaluator",
    "GenerationEvaluator",
    "HallucinationEvaluator",
    "ControlEvaluator",
    "DialogCohesionEvaluator",
    "CIGate",
    "ReportGenerator",
]
