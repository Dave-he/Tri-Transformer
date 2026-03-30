from typing import Union


class TextTokenizer:
    def __init__(self):
        pass

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 127999 for c in text]
