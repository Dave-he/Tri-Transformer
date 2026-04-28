import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings


class ModelService:
    _instance: Optional["ModelService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._model_name = ""
            cls._instance._load_time: Optional[datetime] = None
            cls._instance._model_id: Optional[str] = None
        return cls._instance

    def get_status(self) -> dict:
        return {
            "loaded": self._loaded or settings.mock_inference,
            "model_name": self._model_name or ("mock-inference-v1" if settings.mock_inference else ""),
            "load_time": self._load_time,
        }

    async def load_model(self, model_id: str) -> dict:
        self._model_id = model_id
        self._loaded = False
        task_id = str(uuid.uuid4())
        if settings.mock_inference:
            self._loaded = True
            self._model_name = model_id
            self._load_time = datetime.now(timezone.utc)
        return {"task_id": task_id, "status": "loading"}

    def get_info(self) -> dict:
        model_name = self._model_name or ("mock-inference-v1" if settings.mock_inference else "unknown")
        return {
            "model_name": model_name,
            "version": "1.0.0",
            "parameters_count": 0 if settings.mock_inference else 8000000000,
            "model_type": "mock" if settings.mock_inference else "tri-transformer",
        }


def get_model_service() -> ModelService:
    return ModelService()
