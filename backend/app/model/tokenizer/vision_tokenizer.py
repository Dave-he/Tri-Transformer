import hashlib
import io


VISION_OFFSET = 135000
VISION_RANGE = 10000
CHUNK_SIZE = 64


class VisionTokenizer:
    def __init__(self):
        pass

    def _to_bytes(self, image) -> bytes:
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                return buf.getvalue()
        except ImportError:
            pass
        if isinstance(image, (bytes, bytearray)):
            return bytes(image)
        return str(image).encode()

    def encode(self, images) -> list[int]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        tokens = []
        for image in images:
            raw = self._to_bytes(image)
            if not raw:
                raw = bytes(1)
            for i in range(0, len(raw), CHUNK_SIZE):
                chunk = raw[i: i + CHUNK_SIZE]
                digest = hashlib.md5(chunk).digest()
                token_id = int.from_bytes(digest[:4], "big") % VISION_RANGE + VISION_OFFSET
                tokens.append(token_id)
        return tokens
