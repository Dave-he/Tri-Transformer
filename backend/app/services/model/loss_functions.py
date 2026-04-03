"""
Tri-Transformer 自定义损失函数

包含：
- Hallucination Loss: 幻觉检测损失
- RAG Loss: 检索增强生成损失
- Control Alignment Loss: 控制信号对齐损失
- Total Loss: 综合损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HallucinationLoss(nn.Module):
    """
    幻觉检测损失：通过对比学习区分真实事实与幻觉内容。
    
    对于每个样本，提供：
    - positive_span: 来自知识库的真实内容编码
    - negative_span: 模型生成的潜在幻觉内容编码
    
    损失目标：拉近 positive_span 与上下文编码，推远 negative_span
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        context_enc: torch.Tensor,
        positive_span: torch.Tensor,
        negative_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        context_enc    : [B, D] 上下文编码（来自 I-Transformer）
        positive_span  : [B, D] 真实内容编码
        negative_span  : [B, D] 幻觉内容编码
        """
        pos_dist = F.cosine_similarity(context_enc, positive_span)
        neg_dist = F.cosine_similarity(context_enc, negative_span)
        
        loss_values = torch.relu(self.margin - pos_dist + neg_dist)
        
        if self.reduction == "mean":
            return loss_values.mean()
        elif self.reduction == "sum":
            return loss_values.sum()
        return loss_values


class RAGLoss(nn.Module):
    """
    RAG 检索增强损失：优化检索器与生成器的联合训练。
    
    包含两个组件：
    1. Retrieval Loss: 检索相关性损失（对比学习）
    2. Generation Loss: 生成质量损失（交叉熵）
    """
    
    def __init__(
        self,
        retrieval_weight: float = 0.3,
        generation_weight: float = 0.7,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(
        self,
        query_enc: torch.Tensor,
        doc_encs: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        relevant_doc_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        query_enc     : [B, D] 查询编码
        doc_encs      : [B, K, D] K 个候选文档编码
        logits        : [B, T, V] 生成 logits
        target_ids    : [B, T] 目标 token ids
        relevant_doc_idx: [B] 相关文档索引（用于 retrieval loss）
        
        返回：
        {
            "total_loss": torch.Tensor,
            "retrieval_loss": torch.Tensor,
            "generation_loss": torch.Tensor,
        }
        """
        B, K, D = doc_encs.shape
        
        retrieval_loss = torch.tensor(0.0, device=query_enc.device)
        if relevant_doc_idx is not None:
            query_doc_sim = torch.einsum("bd,bkd->bk", query_enc, doc_encs) / self.temperature
            labels = torch.zeros(B, dtype=torch.long, device=query_enc.device)
            if relevant_doc_idx is not None:
                labels = relevant_doc_idx
            retrieval_loss = F.cross_entropy(query_doc_sim, labels)
        
        generation_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        total_loss = (
            self.retrieval_weight * retrieval_loss +
            self.generation_weight * generation_loss
        )
        
        return {
            "total_loss": total_loss,
            "retrieval_loss": retrieval_loss,
            "generation_loss": generation_loss,
        }


class ControlAlignmentLoss(nn.Module):
    """
    控制信号对齐损失：确保 C-Transformer 生成的控制信号
    与期望的控制模式对齐。
    
    控制信号用于调节 I/O 分支的行为，包括：
    - Thinking Mode: 是否启用深度推理
    - Response Style: 简洁/详细
    - Knowledge Grounding: 知识依赖程度
    """
    
    def __init__(
        self,
        control_dim: int = 4096,
        num_control_modes: int = 4,
    ):
        super().__init__()
        self.control_proj = nn.Linear(control_dim, num_control_modes)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        ctrl_signal: torch.Tensor,
        target_mode: torch.Tensor,
    ) -> torch.Tensor:
        """
        ctrl_signal : [B, D] C-Transformer 生成的控制信号
        target_mode : [B] 目标控制模式标签
        
        返回：控制模式分类损失
        """
        mode_logits = self.control_proj(ctrl_signal)
        return self.ce_loss(mode_logits, target_mode)


class TotalLoss(nn.Module):
    """
    综合损失：加权组合所有损失组件。
    
    总损失 = α * LLM_loss + β * Hallucination_loss + γ * RAG_loss + δ * Control_loss
    """
    
    def __init__(
        self,
        llm_weight: float = 1.0,
        hallucination_weight: float = 0.5,
        rag_weight: float = 0.3,
        control_weight: float = 0.2,
    ):
        super().__init__()
        self.llm_weight = llm_weight
        self.hallucination_weight = hallucination_weight
        self.rag_weight = rag_weight
        self.control_weight = control_weight
        
        self.hallucination_loss = HallucinationLoss()
        self.rag_loss = RAGLoss()
        self.control_loss = ControlAlignmentLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        context_enc: Optional[torch.Tensor] = None,
        positive_span: Optional[torch.Tensor] = None,
        negative_span: Optional[torch.Tensor] = None,
        query_enc: Optional[torch.Tensor] = None,
        doc_encs: Optional[torch.Tensor] = None,
        ctrl_signal: Optional[torch.Tensor] = None,
        target_mode: Optional[torch.Tensor] = None,
        relevant_doc_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        计算综合损失。
        
        必需参数：
        - logits: [B, T, V] 模型输出 logits
        - target_ids: [B, T] 目标 token ids
        
        可选参数（对应不同损失组件）：
        - context_enc, positive_span, negative_span: Hallucination Loss
        - query_enc, doc_encs, relevant_doc_idx: RAG Loss
        - ctrl_signal, target_mode: Control Loss
        
        返回：
        {
            "total_loss": torch.Tensor,
            "llm_loss": torch.Tensor,
            "hallucination_loss": torch.Tensor or None,
            "rag_loss": torch.Tensor or None,
            "control_loss": torch.Tensor or None,
        }
        """
        B, T, V = logits.shape
        llm_loss = self.ce_loss(logits.view(B * T, V), target_ids.view(B * T))
        
        total_loss = self.llm_weight * llm_loss
        
        hall_loss = None
        if context_enc is not None and positive_span is not None and negative_span is not None:
            hall_loss = self.hallucination_loss(context_enc, positive_span, negative_span)
            total_loss = total_loss + self.hallucination_weight * hall_loss
        
        rag_loss_dict = None
        if query_enc is not None and doc_encs is not None:
            rag_loss_dict = self.rag_loss(
                query_enc, doc_encs, logits, target_ids, relevant_doc_idx
            )
            total_loss = total_loss + self.rag_weight * rag_loss_dict["retrieval_loss"]
        
        ctrl_loss = None
        if ctrl_signal is not None and target_mode is not None:
            ctrl_loss = self.control_loss(ctrl_signal, target_mode)
            total_loss = total_loss + self.control_weight * ctrl_loss
        
        return {
            "total_loss": total_loss,
            "llm_loss": llm_loss,
            "hallucination_loss": hall_loss,
            "rag_loss": rag_loss_dict["retrieval_loss"] if rag_loss_dict else None,
            "control_loss": ctrl_loss,
        }
