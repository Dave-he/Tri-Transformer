import random
from typing import Optional

from app.services.model.inference_service import InferenceService


MOCK_RESPONSES = [
    "Based on the provided context, Tri-Transformer uses a three-branch architecture consisting of I-Transformer (encoder), C-Transformer (controller), and O-Transformer (decoder).",
    "The RAG system retrieves relevant documents from the knowledge base and provides them as context for the generation process.",
    "The control branch (C-Transformer) generates control signals that constrain both the encoding and generation branches to ensure knowledge consistency.",
    "This is a mock response generated for development and testing purposes. The actual model would provide a more contextually relevant answer.",
]


class MockInferenceService(InferenceService):
    async def _do_inference(
        self,
        query: str,
        context: list[str],
        history: list[dict],
    ) -> dict:
        response_text = random.choice(MOCK_RESPONSES)
        if context:
            response_text = f"{response_text} [Sources: {len(context)} document(s) referenced.]"
        return {
            "text": response_text,
            "confidence": round(random.uniform(0.75, 0.98), 3),
            "model": "mock-inference-v1",
        }


def get_inference_service() -> InferenceService:
    from app.core.config import settings
    if settings.mock_inference:
        return MockInferenceService()
    from app.services.model.tri_transformer_inference import TriTransformerInferenceService
    return TriTransformerInferenceService()
