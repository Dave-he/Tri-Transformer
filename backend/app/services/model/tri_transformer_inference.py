from typing import Optional

from app.services.model.inference_service import InferenceService, InferenceError


class TriTransformerInferenceService(InferenceService):
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from app.model.tri_transformer import TriTransformerModel, TriTransformerConfig
            from app.model.tokenizer.text_tokenizer import TextTokenizer

            config = TriTransformerConfig()
            model = TriTransformerModel(config)
            if self.model_path:
                state = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state)
            model.eval()
            self._model = model
            self._tokenizer = TextTokenizer()
        except Exception as e:
            raise InferenceError(f"Failed to load TriTransformer model: {e}") from e

    async def _do_inference(
        self,
        query: str,
        context: list[str],
        history: list[dict],
    ) -> dict:
        try:
            import torch

            self._load_model()
            context_text = " ".join(context) if context else ""
            full_input = f"{context_text} {query}".strip() if context_text else query
            input_ids = self._tokenizer.encode(full_input[:512])
            src = torch.tensor([input_ids[:64]], dtype=torch.long)
            tgt = torch.tensor([input_ids[:32]], dtype=torch.long)

            with torch.no_grad():
                output = self._model(src, tgt)
                logits = output.logits
                token_ids = logits.argmax(dim=-1)[0].tolist()
                text = "".join(chr(min(max(t, 32), 126)) for t in token_ids[:100])

            return {
                "text": text,
                "confidence": 0.85,
                "model": "tri-transformer-v1",
            }
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"TriTransformer inference failed: {e}") from e
