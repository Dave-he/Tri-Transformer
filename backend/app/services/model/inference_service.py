import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class InferenceError(Exception):
    pass


class InferenceService(ABC):
    MAX_RETRIES = 3
    BASE_DELAY = 0.5

    async def infer(
        self,
        query: str,
        context: list[str],
        history: Optional[list[dict]] = None,
    ) -> dict:
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return await self._do_inference(query, context, history or [])
            except Exception as e:
                last_exc = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
        raise InferenceError(f"Inference failed after {self.MAX_RETRIES} retries: {last_exc}")

    @abstractmethod
    async def _do_inference(
        self,
        query: str,
        context: list[str],
        history: list[dict],
    ) -> dict:
        pass
