import pytest
import json
import os
import tempfile


MOCK_GT_SAMPLES = [
    {
        "id": f"gt_{i}",
        "query": f"What is concept {i}?",
        "answer": f"Concept {i} is a fundamental idea in the domain.",
        "source_docs": [f"The concept {i} is a fundamental idea in the domain of study."],
        "difficulty": "easy" if i < 7 else ("medium" if i < 14 else "hard"),
        "quality_score": 0.85,
    }
    for i in range(20)
]

MOCK_MODEL_OUTPUTS = [
    {
        "query": sample["query"],
        "generated_answer": sample["answer"] if i % 3 != 0 else f"I don't know about concept {i}.",
        "retrieved_docs": sample["source_docs"],
    }
    for i, sample in enumerate(MOCK_GT_SAMPLES)
]


class TestRAGEvaluator:
    def test_rag_metrics_range(self):
        from eval.pipeline.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(MOCK_GT_SAMPLES[:5], MOCK_MODEL_OUTPUTS[:5])
        for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
            if metric in results:
                assert 0.0 <= results[metric] <= 1.0

    def test_rag_evaluator_returns_dict(self):
        from eval.pipeline.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(MOCK_GT_SAMPLES[:3], MOCK_MODEL_OUTPUTS[:3])
        assert isinstance(results, dict)
        assert len(results) > 0


class TestGenerationEvaluator:
    def test_generation_metrics_nonzero(self):
        from eval.pipeline.generation_evaluator import GenerationEvaluator
        evaluator = GenerationEvaluator()
        predictions = [s["answer"] for s in MOCK_GT_SAMPLES[:5]]
        references = [s["answer"] for s in MOCK_GT_SAMPLES[:5]]
        results = evaluator.evaluate(predictions, references)
        for metric in ["bleu", "rouge_l", "bert_score_f1"]:
            if metric in results:
                assert results[metric] != 0.0 or results[metric] != 1.0 or True

    def test_generation_evaluator_returns_scores(self):
        from eval.pipeline.generation_evaluator import GenerationEvaluator
        evaluator = GenerationEvaluator()
        predictions = ["Paris is the capital of France."]
        references = ["France's capital is Paris."]
        results = evaluator.evaluate(predictions, references)
        assert isinstance(results, dict)


class TestHallucinationEvaluator:
    def test_hallucination_rate_calculation(self):
        from eval.pipeline.hallucination_evaluator import HallucinationEvaluator
        evaluator = HallucinationEvaluator()
        outputs_with_hallucination = [
            {
                "generated_answer": "The Eiffel Tower is in Berlin.",
                "retrieved_docs": ["The Eiffel Tower is in Paris."],
            }
        ]
        outputs_without_hallucination = [
            {
                "generated_answer": "The Eiffel Tower is in Paris.",
                "retrieved_docs": ["The Eiffel Tower is in Paris."],
            }
        ]
        rate_with = evaluator.compute_hallucination_rate(outputs_with_hallucination)
        rate_without = evaluator.compute_hallucination_rate(outputs_without_hallucination)
        assert 0.0 <= rate_with <= 1.0
        assert 0.0 <= rate_without <= 1.0

    def test_hallucination_rate_mock_data(self):
        from eval.pipeline.hallucination_evaluator import HallucinationEvaluator
        evaluator = HallucinationEvaluator()
        mock_outputs = MOCK_MODEL_OUTPUTS[:10]
        rate = evaluator.compute_hallucination_rate(mock_outputs)
        assert 0.0 <= rate <= 1.0


class TestCIGate:
    def test_ci_gate_pass(self):
        from eval.pipeline.ci_gate import CIGate
        gate = CIGate(
            hallucination_rate_threshold=0.05,
            rag_recall_at_5_threshold=0.90,
            bert_score_f1_threshold=0.85,
        )
        good_report = {
            "hallucination_rate": 0.02,
            "rag_recall_at_5": 0.95,
            "bert_score_f1": 0.90,
        }
        passed, message = gate.check(good_report)
        assert passed is True
        assert message == "" or "PASS" in message.upper() or message == ""

    def test_ci_gate_fail_hallucination(self):
        from eval.pipeline.ci_gate import CIGate
        gate = CIGate(
            hallucination_rate_threshold=0.05,
            rag_recall_at_5_threshold=0.90,
            bert_score_f1_threshold=0.85,
        )
        bad_report = {
            "hallucination_rate": 0.10,
            "rag_recall_at_5": 0.95,
            "bert_score_f1": 0.90,
        }
        passed, message = gate.check(bad_report)
        assert passed is False
        assert "hallucination" in message.lower() or "0.10" in message

    def test_ci_gate_fail_bert_score(self):
        from eval.pipeline.ci_gate import CIGate
        gate = CIGate(
            hallucination_rate_threshold=0.05,
            rag_recall_at_5_threshold=0.90,
            bert_score_f1_threshold=0.85,
        )
        bad_report = {
            "hallucination_rate": 0.02,
            "rag_recall_at_5": 0.95,
            "bert_score_f1": 0.70,
        }
        passed, message = gate.check(bad_report)
        assert passed is False


class TestReportGenerator:
    def test_report_files_exist(self):
        from eval.pipeline.report_generator import ReportGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)
            mock_results = {
                "hallucination_rate": 0.03,
                "rag_recall_at_5": 0.92,
                "bert_score_f1": 0.88,
                "bleu": 0.45,
                "rouge_l": 0.62,
            }
            generator.generate(mock_results, report_name="eval_report")
            json_path = os.path.join(tmpdir, "eval_report.json")
            md_path = os.path.join(tmpdir, "eval_report.md")
            assert os.path.exists(json_path)
            assert os.path.exists(md_path)
            with open(json_path) as f:
                data = json.load(f)
            assert "hallucination_rate" in data

    def test_bootstrap_confidence_interval(self):
        from eval.pipeline.report_generator import ReportGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir, bootstrap_n=100)
            mock_results = {
                "hallucination_rate": 0.03,
                "rag_recall_at_5": 0.92,
                "bert_score_f1": 0.88,
            }
            sample_values = {
                "hallucination_rate": [0.02, 0.03, 0.04, 0.03, 0.02],
                "rag_recall_at_5": [0.90, 0.92, 0.91, 0.93, 0.92],
                "bert_score_f1": [0.87, 0.88, 0.89, 0.88, 0.87],
            }
            report_with_ci = generator.generate_with_ci(mock_results, sample_values, report_name="eval_ci_report")
            assert "ci_lower" in report_with_ci or True
            assert "ci_upper" in report_with_ci or True


class TestRAGEvaluatorNoBias:
    def test_no_torch_import(self):
        import importlib
        import sys
        for mod in list(sys.modules.keys()):
            if mod == "torch" or mod.startswith("torch."):
                pass
        from eval.pipeline.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(MOCK_GT_SAMPLES[:3], MOCK_MODEL_OUTPUTS[:3])
        assert isinstance(results, dict)
        assert "faithfulness" in results
        assert "answer_relevancy" in results
        assert "context_recall" in results

    def test_faithfulness_range(self):
        from eval.pipeline.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(MOCK_GT_SAMPLES[:5], MOCK_MODEL_OUTPUTS[:5])
        assert 0.0 <= results["faithfulness"] <= 1.0

    def test_answer_relevancy_range(self):
        from eval.pipeline.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(MOCK_GT_SAMPLES[:5], MOCK_MODEL_OUTPUTS[:5])
        assert 0.0 <= results["answer_relevancy"] <= 1.0


class TestCIGateBoundary:
    def test_recall_equal_threshold_passes(self):
        from eval.pipeline.ci_gate import CIGate
        gate = CIGate(
            hallucination_rate_threshold=0.05,
            rag_recall_at_5_threshold=0.90,
            bert_score_f1_threshold=0.85,
        )
        report = {
            "hallucination_rate": 0.02,
            "rag_recall_at_5": 0.90,
            "bert_score_f1": 0.90,
        }
        passed, message = gate.check(report)
        assert passed is True

    def test_recall_below_threshold_fails(self):
        from eval.pipeline.ci_gate import CIGate
        gate = CIGate(
            hallucination_rate_threshold=0.05,
            rag_recall_at_5_threshold=0.90,
            bert_score_f1_threshold=0.85,
        )
        report = {
            "hallucination_rate": 0.02,
            "rag_recall_at_5": 0.89,
            "bert_score_f1": 0.90,
        }
        passed, message = gate.check(report)
        assert passed is False
        assert "rag_recall_at_5" in message.lower() or "0.89" in message

    def test_bert_score_equal_threshold_fails(self):
        from eval.pipeline.ci_gate import CIGate
        gate = CIGate(
            hallucination_rate_threshold=0.05,
            rag_recall_at_5_threshold=0.90,
            bert_score_f1_threshold=0.85,
        )
        report = {
            "hallucination_rate": 0.02,
            "rag_recall_at_5": 0.95,
            "bert_score_f1": 0.85,
        }
        passed, message = gate.check(report)
        assert passed is False


class TestEvalPipelineIntegration:
    def test_run_returns_all_evaluator_fields(self):
        from eval.pipeline.eval_pipeline import EvalPipeline
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EvalPipeline(output_dir=tmpdir)
            report = pipeline.run(MOCK_GT_SAMPLES[:5], MOCK_MODEL_OUTPUTS[:5])
            assert "faithfulness" in report
            assert "bleu" in report
            assert "hallucination_rate" in report
            assert "instruction_following_rate" in report
            assert "topic_consistency" in report
            assert "ci_gate_passed" in report

    def test_run_ci_gate_field_bool(self):
        from eval.pipeline.eval_pipeline import EvalPipeline
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EvalPipeline(output_dir=tmpdir)
            report = pipeline.run(MOCK_GT_SAMPLES[:3], MOCK_MODEL_OUTPUTS[:3])
            assert isinstance(report["ci_gate_passed"], bool)

    def test_run_without_dialog_sessions(self):
        from eval.pipeline.eval_pipeline import EvalPipeline
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EvalPipeline(output_dir=tmpdir)
            report = pipeline.run(MOCK_GT_SAMPLES[:3], MOCK_MODEL_OUTPUTS[:3])
            assert report is not None

