import asyncio
from typing import AsyncGenerator, Optional


class StreamingEngine:
    def __init__(self, mock_response: Optional[str] = None):
        self.mock_response = mock_response or "这是一个流式生成的示例回复，用于测试 WebSocket 推流功能。"

    async def generate(
        self,
        query: str,
        context: list[str] = None,
        history: list[dict] = None,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        for char in self.mock_response:
            if interrupt_event and interrupt_event.is_set():
                return
            yield char
            await asyncio.sleep(0.005)
