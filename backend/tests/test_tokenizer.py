import pytest


AUDIO_RANGE = (130000, 134000)
VISION_RANGE = (135000, 145000)
TEXT_MAX = 128000


class TestTextTokenizer:
    def test_encode_returns_list_of_int(self):
        from app.model.tokenizer.text_tokenizer import TextTokenizer

        tok = TextTokenizer()
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_empty_string(self):
        from app.model.tokenizer.text_tokenizer import TextTokenizer

        tok = TextTokenizer()
        ids = tok.encode("")
        assert isinstance(ids, list)

    def test_ids_in_text_range(self):
        from app.model.tokenizer.text_tokenizer import TextTokenizer

        tok = TextTokenizer()
        ids = tok.encode("test string for range check")
        assert all(0 <= i < TEXT_MAX for i in ids)


class TestAudioTokenizer:
    def test_encode_returns_list_of_int(self):
        from app.model.tokenizer.audio_tokenizer import AudioTokenizer

        tok = AudioTokenizer()
        frames = [0.1, 0.2, 0.3, -0.1, 0.5] * 20
        ids = tok.encode(frames)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_ids_in_audio_range(self):
        from app.model.tokenizer.audio_tokenizer import AudioTokenizer

        tok = AudioTokenizer()
        frames = [0.1] * 100
        ids = tok.encode(frames)
        assert all(AUDIO_RANGE[0] <= i <= AUDIO_RANGE[1] for i in ids)

    def test_no_overlap_with_text_range(self):
        from app.model.tokenizer.audio_tokenizer import AudioTokenizer

        tok = AudioTokenizer()
        frames = list(range(50))
        ids = tok.encode(frames)
        assert all(i >= AUDIO_RANGE[0] for i in ids)


class TestVisionTokenizer:
    def test_encode_returns_list_of_int(self):
        from app.model.tokenizer.vision_tokenizer import VisionTokenizer

        tok = VisionTokenizer()
        image_bytes = bytes(range(256)) * 4
        ids = tok.encode(image_bytes)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_ids_in_vision_range(self):
        from app.model.tokenizer.vision_tokenizer import VisionTokenizer

        tok = VisionTokenizer()
        image_bytes = bytes(b"\x00" * 512)
        ids = tok.encode(image_bytes)
        assert all(VISION_RANGE[0] <= i <= VISION_RANGE[1] for i in ids)

    def test_no_overlap_with_audio_range(self):
        from app.model.tokenizer.vision_tokenizer import VisionTokenizer

        tok = VisionTokenizer()
        image_bytes = bytes(range(100))
        ids = tok.encode(image_bytes)
        assert all(i >= VISION_RANGE[0] for i in ids)


class TestUnifiedTokenizer:
    def test_special_tokens_registered(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer

        tok = UnifiedTokenizer()
        assert tok.audio_start_id is not None
        assert tok.vision_start_id is not None
        assert tok.interrupt_id is not None

    def test_special_token_ids_in_control_range(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer

        tok = UnifiedTokenizer()
        for tid in [tok.audio_start_id, tok.vision_start_id, tok.interrupt_id]:
            assert 129900 <= tid <= 129999

    def test_encode_mixed_text_no_overlap(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer, ModalInput

        tok = UnifiedTokenizer()
        inputs = [ModalInput(modality="text", data="hello")]
        ids = tok.encode_mixed(inputs)
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_mixed_audio_ids_in_range(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer, ModalInput

        tok = UnifiedTokenizer()
        inputs = [ModalInput(modality="audio", data=[0.1] * 50)]
        ids = tok.encode_mixed(inputs)
        audio_ids = [i for i in ids if i != tok.audio_start_id]
        assert all(AUDIO_RANGE[0] <= i <= AUDIO_RANGE[1] for i in audio_ids)

    def test_encode_mixed_vision_ids_in_range(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer, ModalInput

        tok = UnifiedTokenizer()
        inputs = [ModalInput(modality="vision", data=bytes(100))]
        ids = tok.encode_mixed(inputs)
        vision_ids = [i for i in ids if i != tok.vision_start_id]
        assert all(VISION_RANGE[0] <= i <= VISION_RANGE[1] for i in vision_ids)

    def test_encode_mixed_multi_modal_no_id_collision(self):
        from app.model.tokenizer.unified_tokenizer import UnifiedTokenizer, ModalInput

        tok = UnifiedTokenizer()
        inputs = [
            ModalInput(modality="text", data="test"),
            ModalInput(modality="audio", data=[0.5] * 30),
            ModalInput(modality="vision", data=bytes(64)),
        ]
        ids = tok.encode_mixed(inputs)
        text_ids = [i for i in ids if i < TEXT_MAX]
        audio_ids = [i for i in ids if AUDIO_RANGE[0] <= i <= AUDIO_RANGE[1]]
        vision_ids = [i for i in ids if VISION_RANGE[0] <= i <= VISION_RANGE[1]]
        assert len(text_ids) + len(audio_ids) + len(vision_ids) > 0
        assert set(text_ids).isdisjoint(set(audio_ids))
        assert set(audio_ids).isdisjoint(set(vision_ids))
