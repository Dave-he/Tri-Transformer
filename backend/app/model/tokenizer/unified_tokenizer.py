from dataclasses import dataclass
from typing import Union

from app.model.tokenizer.text_tokenizer import TextTokenizer
from app.model.tokenizer.audio_tokenizer import AudioTokenizer
from app.model.tokenizer.vision_tokenizer import VisionTokenizer


CONTROL_OFFSET = 129900

SPECIAL_TOKENS = {
    "<|audio_start|>": CONTROL_OFFSET,
    "<|vision_start|>": CONTROL_OFFSET + 1,
    "<|interrupt|>": CONTROL_OFFSET + 2,
}


@dataclass
class ModalInput:
    modality: str
    data: Union[str, list[float], bytes]


class UnifiedTokenizer:
    def __init__(self):
        self.text_tokenizer = TextTokenizer()
        self.audio_tokenizer = AudioTokenizer()
        self.vision_tokenizer = VisionTokenizer()
        self._special = SPECIAL_TOKENS

    @property
    def audio_start_id(self) -> int:
        return self._special["<|audio_start|>"]

    @property
    def vision_start_id(self) -> int:
        return self._special["<|vision_start|>"]

    @property
    def interrupt_id(self) -> int:
        return self._special["<|interrupt|>"]

    def get_special_token_id(self, token: str) -> int:
        return self._special[token]

    def encode_mixed(self, inputs: list[ModalInput]) -> list[int]:
        result = []
        for inp in inputs:
            if inp.modality == "text":
                result.extend(self.text_tokenizer.encode(inp.data))
            elif inp.modality == "audio":
                result.append(self.audio_start_id)
                result.extend(self.audio_tokenizer.encode(inp.data))
            elif inp.modality == "vision":
                result.append(self.vision_start_id)
                result.extend(self.vision_tokenizer.encode(inp.data))
        return result
