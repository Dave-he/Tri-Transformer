from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    DOCUMENT_QA = "document_qa"
    DUAL_MODEL = "dual_model"
    KG_TRIPLE = "kg_triple"
    HUMAN = "human"


class GroundTruthItem(BaseModel):
    id: str
    query: str
    answer: str
    source_docs: List[str] = Field(default_factory=list)
    difficulty: str = "easy"
    quality_score: float = 0.5
    source_type: SourceType = SourceType.DOCUMENT_QA
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroundTruthDataset(BaseModel):
    version: str
    items: List[GroundTruthItem]
    stats: Dict[str, Any] = Field(default_factory=dict)
