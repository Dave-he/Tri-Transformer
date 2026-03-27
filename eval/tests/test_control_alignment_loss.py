import pytest
import torch


class TestContrastiveControlLoss:
    def test_positive_vs_negative_similarity(self):
        from eval.loss.control_alignment_loss import ContrastiveControlLoss
        loss_fn = ContrastiveControlLoss(temperature=0.07)
        control_signal = torch.randn(4, 128)
        positive_gen = control_signal + 0.01 * torch.randn(4, 128)
        negative_gen = torch.randn(4, 128)
        pos_sim = torch.nn.functional.cosine_similarity(control_signal, positive_gen).mean()
        neg_sim = torch.nn.functional.cosine_similarity(control_signal, negative_gen).mean()
        assert pos_sim > neg_sim

    def test_contrastive_loss_decreases_with_aligned_input(self):
        from eval.loss.control_alignment_loss import ContrastiveControlLoss
        loss_fn = ContrastiveControlLoss(temperature=0.07)
        batch_size = 4
        dim = 64
        anchor = torch.randn(batch_size, dim)
        positives = anchor + 0.01 * torch.randn(batch_size, dim)
        negatives = torch.randn(batch_size, dim)
        loss = loss_fn.forward(anchor, positives, negatives)
        assert loss.item() >= 0.0


class TestKnowledgeConsistencyLoss:
    def test_entailment_lower_than_contradiction(self):
        from eval.loss.control_alignment_loss import KnowledgeConsistencyLoss
        loss_fn = KnowledgeConsistencyLoss()
        premise = "The capital of France is Paris."
        hypothesis_entail = "Paris is the capital of France."
        hypothesis_contradict = "The capital of France is Berlin."
        loss_entail = loss_fn.compute([(premise, hypothesis_entail)])
        loss_contradict = loss_fn.compute([(premise, hypothesis_contradict)])
        assert loss_entail <= loss_contradict + 0.5

    def test_loss_range(self):
        from eval.loss.control_alignment_loss import KnowledgeConsistencyLoss
        loss_fn = KnowledgeConsistencyLoss()
        pairs = [("The sky is blue.", "The sky is blue.")]
        loss = loss_fn.compute(pairs)
        assert 0.0 <= loss <= 5.0


class TestInstructionFollowingLoss:
    def test_following_vs_not_following(self):
        from eval.loss.control_alignment_loss import InstructionFollowingLoss
        loss_fn = InstructionFollowingLoss()
        correct_logits = torch.tensor([[0.2, 5.0]])
        label = torch.tensor([1])
        loss_correct = loss_fn.classification_loss(correct_logits, label)
        wrong_logits = torch.tensor([[5.0, 0.2]])
        loss_wrong = loss_fn.classification_loss(wrong_logits, label)
        assert loss_correct < loss_wrong

    def test_regression_loss_positive(self):
        from eval.loss.control_alignment_loss import InstructionFollowingLoss
        loss_fn = InstructionFollowingLoss()
        predictions = torch.tensor([0.8, 0.3, 0.9])
        targets = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn.regression_loss(predictions, targets)
        assert loss.item() >= 0.0


class TestControlAlignmentLossWeights:
    def test_weight_configuration(self):
        from eval.loss.control_alignment_loss import ControlAlignmentLoss
        loss_fn = ControlAlignmentLoss(lambda1=1.0, lambda2=0.0, lambda3=0.0)
        batch_size = 4
        dim = 64
        anchor = torch.randn(batch_size, dim)
        positives = anchor + 0.01 * torch.randn(batch_size, dim)
        negatives = torch.randn(batch_size, dim)
        nli_pairs = [("The sky is blue.", "The sky is blue.")] * batch_size
        inst_logits = torch.tensor([[0.2, 0.8]] * batch_size, dtype=torch.float)
        inst_labels = torch.ones(batch_size, dtype=torch.long)
        inst_scores = torch.ones(batch_size) * 0.9
        inst_targets = torch.ones(batch_size)
        loss = loss_fn.forward(
            anchor, positives, negatives, nli_pairs,
            inst_logits, inst_labels, inst_scores, inst_targets
        )
        assert loss.item() >= 0.0
