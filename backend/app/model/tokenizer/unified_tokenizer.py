from dataclasses import dataclass
from typing import Any, Union

from app.model.tokenizer.text_tokenizer import TextTokenizer
from app.model.tokenizer.audio_tokenizer import AudioTokenizer
from app.model.tokenizer.vision_tokenizer import VisionTokenizer


@dataclass
class ModalInput:
    modality: str
    data: Any


CONTROL_OFFSET = 129900

SPECIAL_TOKENS = {
    "<|audio_start|>": CONTROL_OFFSET,
    "<|vision_start|>": CONTROL_OFFSET + 1,
    "<|interrupt|>": CONTROL_OFFSET + 2,
}


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

    def token_to_id(self, token: str) -> Union[int, None]:
        return self._special.get(token)

    def get_special_token_id(self, token: str) -> int:
        return self._special[token]

    def encode_mixed(self, inputs: list) -> list[int]:
        result = []
        for inp in inputs:
            if hasattr(inp, "modality"):
                modality = inp.modality
                data = inp.data
            elif isinstance(inp, (list, tuple)) and len(inp) == 2:
                modality, data = inp
            else:
                continue

            if modality == "text":
                result.extend(self.text_tokenizer.encode(data))
            elif modality == "audio":
                result.append(self.audio_start_id)
                result.extend(self.audio_tokenizer.encode(data))
            elif modality == "vision":
                result.append(self.vision_start_id)
                result.extend(self.vision_tokenizer.encode(data))
        return result
