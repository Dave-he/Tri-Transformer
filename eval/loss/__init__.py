def __getattr__(name):
    _torch_classes = {
        "BaseLoss": ("eval.loss.base", "BaseLoss"),
        "FactualHallucinationLoss": ("eval.loss.hallucination_loss", "FactualHallucinationLoss"),
        "SourceAttributionLoss": ("eval.loss.hallucination_loss", "SourceAttributionLoss"),
        "AbstentionCalibrationLoss": ("eval.loss.hallucination_loss", "AbstentionCalibrationLoss"),
        "HallucinationLoss": ("eval.loss.hallucination_loss", "HallucinationLoss"),
        "RetrievalRelevanceLoss": ("eval.loss.rag_loss", "RetrievalRelevanceLoss"),
        "CoverageLoss": ("eval.loss.rag_loss", "CoverageLoss"),
        "RankingConsistencyLoss": ("eval.loss.rag_loss", "RankingConsistencyLoss"),
        "RAGLoss": ("eval.loss.rag_loss", "RAGLoss"),
        "ContrastiveControlLoss": ("eval.loss.control_alignment_loss", "ContrastiveControlLoss"),
        "KnowledgeConsistencyLoss": ("eval.loss.control_alignment_loss", "KnowledgeConsistencyLoss"),
        "InstructionFollowingLoss": ("eval.loss.control_alignment_loss", "InstructionFollowingLoss"),
        "ControlAlignmentLoss": ("eval.loss.control_alignment_loss", "ControlAlignmentLoss"),
        "TotalLoss": ("eval.loss.total_loss", "TotalLoss"),
    }
    if name in _torch_classes:
        module_path, cls_name = _torch_classes[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module 'eval.loss' has no attribute {name!r}")


__all__ = [
    "BaseLoss",
    "FactualHallucinationLoss",
    "SourceAttributionLoss",
    "AbstentionCalibrationLoss",
    "HallucinationLoss",
    "RetrievalRelevanceLoss",
    "CoverageLoss",
    "RankingConsistencyLoss",
    "RAGLoss",
    "ContrastiveControlLoss",
    "KnowledgeConsistencyLoss",
    "InstructionFollowingLoss",
    "ControlAlignmentLoss",
    "TotalLoss",
]
