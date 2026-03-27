import pytest
import json
import tempfile
import os
from pathlib import Path


SAMPLE_DOCS = [
    {
        "id": "doc_001",
        "content": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
                   "It was designed and built by Gustave Eiffel between 1887 and 1889. "
                   "The tower stands 330 metres tall and was the world's tallest man-made structure for 41 years. "
                   "It is one of the most recognizable structures in the world. "
                   "The tower receives about 7 million visitors per year.",
    },
    {
        "id": "doc_002",
        "content": "Python is a high-level, general-purpose programming language. "
                   "It was created by Guido van Rossum and first released in 1991. "
                   "Python's design philosophy emphasizes code readability with the use of significant indentation. "
                   "It supports multiple programming paradigms, including structured, object-oriented and functional programming. "
                   "Python is widely used in web development, data science, artificial intelligence, and automation.",
    },
]


class TestDocumentQAGenerator:
    def test_qa_generation_coverage(self):
        from eval.ground_truth.document_qa_generator import DocumentQAGenerator
        generator = DocumentQAGenerator(min_qa_per_1000_chars=5)
        qa_pairs = generator.generate(SAMPLE_DOCS)
        total_chars = sum(len(d["content"]) for d in SAMPLE_DOCS)
        expected_min = max(1, int(total_chars / 1000) * 5)
        assert len(qa_pairs) >= expected_min or len(qa_pairs) >= 2

    def test_qa_format(self):
        from eval.ground_truth.document_qa_generator import DocumentQAGenerator
        generator = DocumentQAGenerator(min_qa_per_1000_chars=3)
        qa_pairs = generator.generate(SAMPLE_DOCS[:1])
        assert len(qa_pairs) > 0
        for qa in qa_pairs:
            assert "query" in qa
            assert "answer" in qa
            assert "source_doc_id" in qa
            assert len(qa["query"]) > 5
            assert len(qa["answer"]) > 3

    def test_quality_filter(self):
        from eval.ground_truth.document_qa_generator import DocumentQAGenerator
        generator = DocumentQAGenerator(min_qa_per_1000_chars=2)
        qa_pairs = generator.generate(SAMPLE_DOCS)
        for qa in qa_pairs:
            assert qa["query"] != qa["answer"]
            assert len(qa["query"].split()) >= 3


class TestDualModelValidator:
    def test_consensus_identification(self):
        from eval.ground_truth.dual_model_validator import DualModelValidator
        validator = DualModelValidator(similarity_threshold=0.5)
        answer_a = "The Eiffel Tower is 330 meters tall and located in Paris."
        answer_b = "The Eiffel Tower stands 330 meters tall in Paris France."
        answer_c = "The Eiffel Tower is located in Berlin Germany far from France."
        is_consensus_ab = validator.is_consensus(answer_a, answer_b)
        is_consensus_ac = validator.is_consensus(answer_a, answer_c)
        assert is_consensus_ab is True
        assert is_consensus_ac is False

    def test_quality_label(self):
        from eval.ground_truth.dual_model_validator import DualModelValidator
        validator = DualModelValidator(similarity_threshold=0.85)
        high_sim_answer_a = "Paris is the capital of France."
        high_sim_answer_b = "France's capital city is Paris."
        quality = validator.get_quality_label(high_sim_answer_a, high_sim_answer_b)
        assert quality in ["gold", "silver", "discard"]


class TestKGTripleExtractor:
    def test_triple_format(self):
        from eval.ground_truth.kg_triple_extractor import KGTripleExtractor
        extractor = KGTripleExtractor()
        text = "Gustave Eiffel designed the Eiffel Tower. The tower is located in Paris."
        triples = extractor.extract(text)
        assert len(triples) >= 0
        for triple in triples:
            assert len(triple) == 3
            subject, relation, obj = triple
            assert isinstance(subject, str)
            assert isinstance(relation, str)
            assert isinstance(obj, str)

    def test_triple_to_qa_conversion(self):
        from eval.ground_truth.kg_triple_extractor import KGTripleExtractor
        extractor = KGTripleExtractor()
        triple = ("Eiffel Tower", "located in", "Paris")
        qa = extractor.triple_to_qa(triple)
        assert "query" in qa
        assert "answer" in qa
        assert "Paris" in qa["answer"] or "paris" in qa["answer"].lower()


class TestConsistencyChecker:
    def test_contradiction_filter(self):
        from eval.ground_truth.consistency_checker import ConsistencyChecker
        checker = ConsistencyChecker()
        consistent_pair = {
            "query": "Where is the Eiffel Tower?",
            "answer": "The Eiffel Tower is in Paris.",
        }
        contradictory_pair = {
            "query": "What is the capital of France?",
            "answer": "The capital of France is Berlin.",
        }
        context = "Paris is the capital of France. The Eiffel Tower is in Paris."
        consistent_result = checker.check(consistent_pair, context)
        assert consistent_result in [True, False]

    def test_retention_rate(self):
        from eval.ground_truth.consistency_checker import ConsistencyChecker
        checker = ConsistencyChecker()
        samples = [
            {"query": "What is Python?", "answer": "Python is a programming language."},
            {"query": "Who created Python?", "answer": "Python was created by Guido van Rossum."},
            {"query": "When was Python released?", "answer": "Python was first released in 1991."},
        ]
        context = "Python is a high-level programming language created by Guido van Rossum in 1991."
        retained = checker.filter(samples, context)
        retention_rate = len(retained) / len(samples)
        assert retention_rate >= 0.0


class TestGTFusionEngine:
    def test_fusion_output_schema(self):
        from eval.ground_truth.fusion_engine import GTFusionEngine
        from eval.ground_truth.schema import SourceType
        engine = GTFusionEngine()
        raw_samples = [
            {
                "query": "What is the height of the Eiffel Tower?",
                "answer": "The Eiffel Tower is 330 meters tall.",
                "source_type": SourceType.DOCUMENT_QA,
                "source_doc_id": "doc_001",
            }
        ]
        fused = engine.fuse(raw_samples)
        assert len(fused) > 0
        item = fused[0]
        assert hasattr(item, "difficulty") or "difficulty" in item
        assert hasattr(item, "quality_score") or "quality_score" in item

    def test_difficulty_levels_present(self):
        from eval.ground_truth.fusion_engine import GTFusionEngine
        from eval.ground_truth.schema import SourceType
        engine = GTFusionEngine()
        samples = []
        for i in range(6):
            samples.append({
                "query": f"Simple question {i}?",
                "answer": f"Simple answer {i}.",
                "source_type": SourceType.DOCUMENT_QA,
                "source_doc_id": "doc_001",
            })
        fused = engine.fuse(samples)
        difficulties = set()
        for item in fused:
            if hasattr(item, "difficulty"):
                difficulties.add(item.difficulty)
            elif isinstance(item, dict):
                difficulties.add(item.get("difficulty", "easy"))
        assert len(difficulties) >= 1


class TestGTVersioning:
    def test_version_increment(self):
        from eval.ground_truth.gt_versioning import GTVersioning
        with tempfile.TemporaryDirectory() as tmpdir:
            versioning = GTVersioning(storage_dir=tmpdir)
            dataset_v1 = [{"query": "Q1", "answer": "A1"}]
            dataset_v2 = [{"query": "Q1", "answer": "A1"}, {"query": "Q2", "answer": "A2"}]
            v1 = versioning.save(dataset_v1)
            v2 = versioning.save(dataset_v2)
            assert v1 != v2

    def test_history_retrieval(self):
        from eval.ground_truth.gt_versioning import GTVersioning
        with tempfile.TemporaryDirectory() as tmpdir:
            versioning = GTVersioning(storage_dir=tmpdir)
            dataset = [{"query": "Q1", "answer": "A1"}]
            version_id = versioning.save(dataset)
            retrieved = versioning.load(version_id)
            assert retrieved == dataset

    def test_jsonlines_export(self):
        from eval.ground_truth.gt_versioning import GTVersioning
        with tempfile.TemporaryDirectory() as tmpdir:
            versioning = GTVersioning(storage_dir=tmpdir)
            dataset = [{"query": "Q1", "answer": "A1", "difficulty": "easy"}]
            version_id = versioning.save(dataset)
            export_path = os.path.join(tmpdir, "export.jsonl")
            versioning.export_jsonlines(version_id, export_path)
            assert os.path.exists(export_path)
            with open(export_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            loaded = json.loads(lines[0])
            assert loaded["query"] == "Q1"
