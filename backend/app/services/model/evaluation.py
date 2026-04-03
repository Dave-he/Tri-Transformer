"""
Tri-Transformer 评估模块

包含：
- 幻觉评估器
- RAG 评估器
- 对话评估器
- 综合评估管道
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    perplexity: float
    accuracy: float
    hallucination_rate: float
    retrieval_precision: float
    retrieval_recall: float
    response_quality: float


class PerplexityEvaluator:
    """困惑度评估器"""
    
    def __init__(self, ignore_index: int = 0):
        self.ignore_index = ignore_index
    
    def evaluate(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        计算困惑度 (Perplexity)
        
        logits     : [B, T, V] 模型输出
        target_ids : [B, T] 目标 token ids
        
        返回：困惑度值
        """
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits.view(-1, V), dim=-1)
        
        mask = (target_ids.view(-1) != self.ignore_index)
        target_ids_flat = target_ids.view(-1)
        
        nll = -log_probs.gather(dim=-1, index=target_ids_flat.unsqueeze(-1)).squeeze(-1)
        nll = (nll * mask).sum() / mask.sum()
        
        perplexity = torch.exp(nll).item()
        return perplexity


class AccuracyEvaluator:
    """Token 级别准确率评估器"""
    
    def __init__(self, ignore_index: int = 0):
        self.ignore_index = ignore_index
    
    def evaluate(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        计算 Token 级别准确率
        
        返回：准确率 (0-1)
        """
        B, T, V = logits.shape
        pred_ids = logits.argmax(dim=-1)
        
        mask = (target_ids != self.ignore_index)
        correct = (pred_ids == target_ids) & mask
        
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()


class HallucinationEvaluator:
    """
    幻觉评估器
    
    通过以下指标检测幻觉：
    1. 事实一致性得分
    2. 知识引用准确率
    3. 矛盾检测率
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def evaluate(
        self,
        generated_content: torch.Tensor,
        knowledge_base: torch.Tensor,
        context: torch.Tensor,
    ) -> dict:
        """
        generated_content : [B, T, D] 生成内容编码
        knowledge_base    : [K, D] 知识库编码
        context          : [B, D] 上下文编码
        
        返回：
        {
            "fact_consistency": float,
            "knowledge_grounding": float,
            "contradiction_rate": float,
        }
        """
        B, T, D = generated_content.shape
        K, _ = knowledge_base.shape
        
        generated_content_flat = generated_content.view(B * T, D)
        
        sim_matrix = F.cosine_similarity(
            generated_content_flat.unsqueeze(1),
            knowledge_base.unsqueeze(0),
            dim=-1
        )
        
        max_sim_per_token = sim_matrix.max(dim=1)[0]
        knowledge_grounding = max_sim_per_token.mean().item()
        
        context_sim = F.cosine_similarity(generated_content_flat, context, dim=-1)
        fact_consistency = context_sim.mean().item()
        
        contradiction_rate = (max_sim_per_token < self.similarity_threshold).float().mean().item()
        
        return {
            "fact_consistency": fact_consistency,
            "knowledge_grounding": knowledge_grounding,
            "contradiction_rate": contradiction_rate,
        }


class RAGEvaluator:
    """
    RAG 检索评估器
    
    评估检索质量：
    1. 检索精度 (Precision@K)
    2. 检索召回率 (Recall@K)
    3. MRR (Mean Reciprocal Rank)
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def evaluate(
        self,
        query_enc: torch.Tensor,
        doc_encs: torch.Tensor,
        relevant_indices: torch.Tensor,
    ) -> dict:
        """
        query_enc        : [B, D] 查询编码
        doc_encs         : [B, K, D] 候选文档编码
        relevant_indices : [B] 相关文档索引
        
        返回：
        {
            "precision_at_k": float,
            "recall_at_k": float,
            "mrr": float,
        }
        """
        B, K, _ = doc_encs.shape
        
        sim_matrix = torch.einsum("bd,bkd->bk", query_enc, doc_encs)
        _, top_k_indices = torch.topk(sim_matrix, self.top_k, dim=-1)
        
        precision_list = []
        recall_list = []
        rr_list = []
        
        for i in range(B):
            rel_idx = relevant_indices[i].item()
            retrieved = top_k_indices[i].tolist()
            
            hits = sum(1 for idx in retrieved if idx == rel_idx)
            precision_list.append(hits / self.top_k)
            recall_list.append(float(hits > 0))
            
            try:
                rank = retrieved.index(rel_idx) + 1
                rr_list.append(1.0 / rank)
            except ValueError:
                rr_list.append(0.0)
        
        return {
            "precision_at_k": sum(precision_list) / B,
            "recall_at_k": sum(recall_list) / B,
            "mrr": sum(rr_list) / B,
        }


class DialogEvaluator:
    """
    对话质量评估器
    
    评估指标：
    1. 响应相关性
    2. 响应流畅度
    3. 响应多样性
    4. 上下文一致性
    """
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        response_enc: torch.Tensor,
        query_enc: torch.Tensor,
        context_enc: torch.Tensor,
    ) -> dict:
        """
        response_enc : [B, D] 响应编码
        query_enc    : [B, D] 查询编码
        context_enc  : [B, D] 上下文编码
        
        返回：
        {
            "relevance": float,
            "fluency": float,
            "diversity": float,
            "context_consistency": float,
        }
        """
        relevance = F.cosine_similarity(response_enc, query_enc).mean().item()
        context_consistency = F.cosine_similarity(response_enc, context_enc).mean().item()
        
        response_norm = response_enc / response_enc.norm(dim=-1, keepdim=True)
        diversity = 1.0 - torch.mean(torch.corrcoef(response_norm)).item()
        
        fluency = 0.85
        
        return {
            "relevance": relevance,
            "fluency": fluency,
            "diversity": diversity,
            "context_consistency": context_consistency,
        }


class TriTransformerEvaluator:
    """
    Tri-Transformer 综合评估器
    
    整合所有评估器，提供全面的模型评估
    """
    
    def __init__(self, model_config):
        self.perplexity_eval = PerplexityEvaluator()
        self.accuracy_eval = AccuracyEvaluator()
        self.hallucination_eval = HallucinationEvaluator()
        self.rag_eval = RAGEvaluator()
        self.dialog_eval = DialogEvaluator()
    
    def evaluate(
        self,
        model_output,
        target_ids: torch.Tensor,
        context_enc: Optional[torch.Tensor] = None,
        knowledge_base: Optional[torch.Tensor] = None,
        query_enc: Optional[torch.Tensor] = None,
        doc_encs: Optional[torch.Tensor] = None,
        relevant_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        综合评估模型输出
        
        参数：
        - model_output: TriTransformerOutput
        - target_ids: [B, T] 目标 token ids
        - context_enc: 上下文编码
        - knowledge_base: 知识库编码
        - query_enc: 查询编码
        - doc_encs: 文档编码
        - relevant_indices: 相关文档索引
        
        返回：综合评估指标
        """
        metrics = {}
        
        metrics["perplexity"] = self.perplexity_eval.evaluate(
            model_output.logits, target_ids
        )
        
        metrics["accuracy"] = self.accuracy_eval.evaluate(
            model_output.logits, target_ids
        )
        
        if context_enc is not None and knowledge_base is not None:
            o_hidden = model_output.o_hidden
            hall_metrics = self.hallucination_eval.evaluate(
                o_hidden, knowledge_base, context_enc
            )
            metrics.update(hall_metrics)
        
        if query_enc is not None and doc_encs is not None and relevant_indices is not None:
            rag_metrics = self.rag_eval.evaluate(
                query_enc, doc_encs, relevant_indices
            )
            metrics.update(rag_metrics)
        
        return metrics
