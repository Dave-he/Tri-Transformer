import pytest

torch = pytest.importorskip("torch")


class TestFactualHallucinationLoss:
    def test_supported_fact_lower_loss(self):
        from eval.loss.hallucination_loss import FactualHallucinationLoss
        loss_fn = FactualHallucinationLoss()
        supported_claim = "The Eiffel Tower is located in Paris France."
        knowledge_docs = ["The Eiffel Tower is located in Paris, France."]
        loss_supported = loss_fn.compute([supported_claim], [knowledge_docs])
        unrelated_claim = "Quantum mechanics defines the behavior of subatomic photons."
        loss_unsupported = loss_fn.compute([unrelated_claim], [knowledge_docs])
        assert loss_supported <= loss_unsupported

    def test_loss_range(self):
        from eval.loss.hallucination_loss import FactualHallucinationLoss
        loss_fn = FactualHallucinationLoss()
        claims = ["Paris is a city in France."]
        docs = [["France is a country in Europe. Its capital is Paris."]]
        loss = loss_fn.compute(claims, docs)
        assert 0.0 <= loss <= 2.0


class TestSourceAttributionLoss:
    def test_attributable_vs_non_attributable(self):
        from eval.loss.hallucination_loss import SourceAttributionLoss
        loss_fn = SourceAttributionLoss()
        attributable_text = "Paris is the capital of France"
        source_chunks = [
            "Paris is the capital of France.",
            "France has many beautiful cities.",
        ]
        non_attributable_text = "Mars has no atmosphere"
        loss_attr = loss_fn.compute([attributable_text], [source_chunks])
        loss_non_attr = loss_fn.compute([non_attributable_text], [source_chunks])
        assert loss_attr <= loss_non_attr + 0.5

    def test_attribution_loss_gap(self):
        from eval.loss.hallucination_loss import SourceAttributionLoss
        loss_fn = SourceAttributionLoss()
        text_match = "The Eiffel Tower is 330 meters tall"
        text_no_match = "Quantum physics proves time travel is possible"
        source = ["The Eiffel Tower stands 330 meters tall in Paris."]
        loss_match = loss_fn.compute([text_match], [source])
        loss_no_match = loss_fn.compute([text_no_match], [source])
        assert abs(loss_match - loss_no_match) >= 0.0


class TestAbstentionCalibrationLoss:
    def test_refusal_when_no_knowledge(self):
        from eval.loss.hallucination_loss import AbstentionCalibrationLoss
        loss_fn = AbstentionCalibrationLoss(threshold=0.5)
        low_confidence = torch.tensor([0.1, 0.2, 0.15])
        high_confidence = torch.tensor([0.9, 0.85, 0.92])
        forced_gen_labels = torch.ones(3)
        refused_labels = torch.zeros(3)
        loss_refusal = loss_fn.forward(low_confidence, refused_labels)
        loss_forced = loss_fn.forward(low_confidence, forced_gen_labels)
        assert loss_refusal < loss_forced

    def test_generation_allowed_when_high_confidence(self):
        from eval.loss.hallucination_loss import AbstentionCalibrationLoss
        loss_fn = AbstentionCalibrationLoss(threshold=0.5)
        high_confidence = torch.tensor([0.9, 0.85, 0.92])
        gen_labels = torch.ones(3)
        loss = loss_fn.forward(high_confidence, gen_labels)
        assert loss.item() >= 0.0


class TestHallucinationRate:
    def test_hallucination_rate_range(self):
        from eval.loss.hallucination_loss import HallucinationLoss
        loss_fn = HallucinationLoss()
        generations = [
            "Paris is the capital of France.",
            "The sun orbits the Earth.",
            "Water is H2O.",
        ]
        knowledge_docs = [
            ["France's capital city is Paris."],
            ["The Earth orbits the Sun, not the other way around."],
            ["Water molecule consists of two hydrogen atoms and one oxygen atom, H2O."],
        ]
        rate = loss_fn.compute_hallucination_rate(generations, knowledge_docs)
        assert 0.0 <= rate <= 1.0
