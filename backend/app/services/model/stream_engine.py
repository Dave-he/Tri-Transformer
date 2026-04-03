import asyncio
from typing import AsyncGenerator, Optional


class StreamingEngine:
    def __init__(self, mock_response: Optional[str] = None):
        self.mock_response = mock_response or "这是一个流式生成的示例回复，用于测试 WebSocket 推流功能。"
        self._interrupt_event: Optional[asyncio.Event] = None

    async def generate(
        self,
        query: str,
        context: list[str] = None,
        history: list[dict] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        _interrupt = interrupt_event or self._interrupt_event

        if _interrupt and _interrupt.is_set():
            yield {"token": "", "done": True, "interrupted": True}
            return

        tokens = list(self.mock_response)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]

        for i, char in enumerate(tokens):
            if _interrupt and _interrupt.is_set():
                yield {"token": "", "done": True, "interrupted": True}
                return
            yield {"token": char, "done": False}
            await asyncio.sleep(0.001)

        yield {"token": "", "done": True, "interrupted": False}
