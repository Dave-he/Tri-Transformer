import hashlib
from typing import Union


AUDIO_OFFSET = 130000
AUDIO_RANGE = 4000


class AudioTokenizer:
    def __init__(self, frame_size: int = 10):
        self.frame_size = frame_size

    def encode(self, frames: list[float]) -> list[int]:
        tokens = []
        for i in range(0, max(len(frames), 1), self.frame_size):
            chunk = frames[i : i + self.frame_size]
            digest = hashlib.md5(str(chunk).encode()).digest()
            token_id = int.from_bytes(digest[:4], "big") % AUDIO_RANGE + AUDIO_OFFSET
            tokens.append(token_id)
        return tokens
