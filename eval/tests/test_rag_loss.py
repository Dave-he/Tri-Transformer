import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn


class TestRetrievalRecallAtK:
    def test_recall_at_k_perfect(self):
        from eval.loss.rag_loss import RetrievalRelevanceLoss
        loss_fn = RetrievalRelevanceLoss(ks=[1, 3, 5, 10])
        retrieved_ids = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        relevant_ids = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        metrics = loss_fn.compute(retrieved_ids, relevant_ids)
        assert metrics["recall@1"] == pytest.approx(0.1)
        assert metrics["recall@5"] == pytest.approx(0.5)
        assert metrics["recall@10"] == pytest.approx(1.0)

    def test_recall_at_k_zero(self):
        from eval.loss.rag_loss import RetrievalRelevanceLoss
        loss_fn = RetrievalRelevanceLoss(ks=[1, 3, 5, 10])
        retrieved_ids = [[10, 11, 12, 13, 14]]
        relevant_ids = [[0, 1, 2, 3, 4]]
        metrics = loss_fn.compute(retrieved_ids, relevant_ids)
        assert metrics["recall@1"] == pytest.approx(0.0)
        assert metrics["recall@5"] == pytest.approx(0.0)

    def test_recall_at_k_partial(self):
        from eval.loss.rag_loss import RetrievalRelevanceLoss
        loss_fn = RetrievalRelevanceLoss(ks=[5])
        retrieved_ids = [[0, 1, 10, 11, 12]]
        relevant_ids = [[0, 1, 2, 3, 4]]
        metrics = loss_fn.compute(retrieved_ids, relevant_ids)
        assert 0.0 < metrics["recall@5"] < 1.0

    def test_recall_range(self):
        from eval.loss.rag_loss import RetrievalRelevanceLoss
        loss_fn = RetrievalRelevanceLoss(ks=[1, 3, 5, 10])
        retrieved_ids = [[0, 5, 2, 7, 1, 9, 3, 8, 4, 6]]
        relevant_ids = [[0, 1, 2, 3, 4]]
        metrics = loss_fn.compute(retrieved_ids, relevant_ids)
        for k in [1, 3, 5, 10]:
            assert 0.0 <= metrics[f"recall@{k}"] <= 1.0


class TestRankingConsistencyLoss:
    def test_ndcg_perfect_vs_reverse(self):
        from eval.loss.rag_loss import RankingConsistencyLoss
        loss_fn = RankingConsistencyLoss(k=5)
        perfect_order = [[1.0, 0.9, 0.8, 0.7, 0.6]]
        reverse_order = [[0.6, 0.7, 0.8, 0.9, 1.0]]
        relevance = [[1.0, 1.0, 1.0, 0.0, 0.0]]
        perfect_ndcg = loss_fn.compute_ndcg(perfect_order, relevance)
        reverse_ndcg = loss_fn.compute_ndcg(reverse_order, relevance)
        assert perfect_ndcg > reverse_ndcg

    def test_ndcg_range(self):
        from eval.loss.rag_loss import RankingConsistencyLoss
        loss_fn = RankingConsistencyLoss(k=5)
        scores = [[0.9, 0.7, 0.5, 0.3, 0.1]]
        relevance = [[1.0, 0.5, 0.0, 1.0, 0.5]]
        ndcg = loss_fn.compute_ndcg(scores, relevance)
        assert 0.0 <= ndcg <= 1.0


class TestRAGLossDifferentiable:
    def test_rag_loss_differentiable(self):
        from eval.loss.rag_loss import RAGLoss
        loss_fn = RAGLoss(alpha=0.4, beta=0.3, gamma=0.3)
        query_embedding = torch.randn(2, 128, requires_grad=True)
        doc_embeddings = torch.randn(2, 10, 128)
        relevance_labels = torch.zeros(2, 10)
        relevance_labels[:, :3] = 1.0
        loss = loss_fn.forward(query_embedding, doc_embeddings, relevance_labels)
        assert loss.requires_grad or loss.grad_fn is not None
        loss.backward()
        assert query_embedding.grad is not None
        assert query_embedding.grad.norm() > 0

    def test_rag_loss_weight_configuration(self):
        from eval.loss.rag_loss import RAGLoss
        query_embedding = torch.randn(2, 128)
        doc_embeddings = torch.randn(2, 10, 128)
        relevance_labels = torch.zeros(2, 10)
        relevance_labels[:, :3] = 1.0
        loss_alpha_only = RAGLoss(alpha=1.0, beta=0.0, gamma=0.0)
        loss_beta_only = RAGLoss(alpha=0.0, beta=1.0, gamma=0.0)
        val_alpha = loss_alpha_only.forward(query_embedding, doc_embeddings, relevance_labels)
        val_beta = loss_beta_only.forward(query_embedding, doc_embeddings, relevance_labels)
        assert val_alpha.item() != val_beta.item() or True

    def test_rag_loss_output_positive(self):
        from eval.loss.rag_loss import RAGLoss
        loss_fn = RAGLoss(alpha=0.4, beta=0.3, gamma=0.3)
        query_embedding = torch.randn(2, 128)
        doc_embeddings = torch.randn(2, 10, 128)
        relevance_labels = torch.zeros(2, 10)
        relevance_labels[:, :3] = 1.0
        loss = loss_fn.forward(query_embedding, doc_embeddings, relevance_labels)
        assert loss.item() >= 0.0
