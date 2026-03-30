import hashlib


VISION_OFFSET = 135000
VISION_RANGE = 10000
CHUNK_SIZE = 64


class VisionTokenizer:
    def __init__(self):
        pass

    def encode(self, image_bytes: bytes) -> list[int]:
        if not image_bytes:
            image_bytes = bytes(1)
        tokens = []
        for i in range(0, len(image_bytes), CHUNK_SIZE):
            chunk = image_bytes[i : i + CHUNK_SIZE]
            digest = hashlib.md5(chunk).digest()
            token_id = int.from_bytes(digest[:4], "big") % VISION_RANGE + VISION_OFFSET
            tokens.append(token_id)
        return tokens
