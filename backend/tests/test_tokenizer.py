import pytest
import torch
import numpy as np
from PIL import Image
from io import BytesIO


class TestTextTokenizer:
    def test_encode_returns_list_of_int(self):
        from app.model.tokenizer.text_tokenizer import TextTokenizer
        tokenizer = TextTokenizer(vocab_size=30522)
        tokens = tokenizer.encode("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0


class TestAudioTokenizer:
    def test_audio_token_ids_in_range(self):
        from app.model.tokenizer.audio_tokenizer import AudioTokenizer
        tokenizer = AudioTokenizer()
        audio_frames = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
        tokens = tokenizer.encode(audio_frames)
        assert isinstance(tokens, list)
        assert all(130000 <= t <= 134000 for t in tokens)


class TestVisionTokenizer:
    def test_vision_token_ids_in_range(self):
        from app.model.tokenizer.vision_tokenizer import VisionTokenizer
        tokenizer = VisionTokenizer()
        images = [
            Image.new("RGB", (224, 224), color=(i * 20, i * 10, 128))
            for i in range(3)
        ]
        tokens = tokenizer.encode(images)
        assert isinstance(tokens, list)
        assert all(135000 <= t <= 145000 for t in tokens)


class TestUnifiedTokenizer:
    def test_encode_mixed_no_overflow(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer
        tokenizer = UnifiedTokenizer()
        mixed = [
            ("text", "hello world"),
            ("audio", [np.random.randn(16000).astype(np.float32)]),
            ("vision", [Image.new("RGB", (224, 224), color=(100, 50, 200))]),
        ]
        tokens = tokenizer.encode_mixed(mixed)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        max_text = 30522
        max_audio = 134000
        max_vision = 145000
        assert all(t <= max_vision for t in tokens)

    def test_special_tokens_registered(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer
        tokenizer = UnifiedTokenizer()
        special_tokens = ["<|audio_start|>", "<|vision_start|>", "<|interrupt|>"]
        for tok in special_tokens:
            assert tokenizer.token_to_id(tok) is not None

    def test_token_id_ranges_non_overlapping(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer
        tokenizer = UnifiedTokenizer()
        text_tok = tokenizer.encode_mixed([("text", "hi")])
        audio_tok = tokenizer.encode_mixed([("audio", [np.random.randn(8000).astype(np.float32)])])
        vision_tok = tokenizer.encode_mixed([("vision", [Image.new("RGB", (64, 64))])])
        text_ids = set(text_tok)
        audio_ids = set(audio_tok)
        vision_ids = set(vision_tok)
        assert text_ids.isdisjoint(audio_ids), "Text and audio tokens overlap"
        assert text_ids.isdisjoint(vision_ids), "Text and vision tokens overlap"
        assert audio_ids.isdisjoint(vision_ids), "Audio and vision tokens overlap"
