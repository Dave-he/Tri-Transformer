from unittest.mock import MagicMock, patch

import pytest

from app.model.tokenizer.text_tokenizer import TextTokenizer


def _make_mock_tokenizer():
    mock_tok = MagicMock()
    mock_tok.vocab_size = 151936
    mock_tok.encode.side_effect = lambda text, **kw: ([] if not text else [9906, 1917])
    mock_tok.decode.return_value = "hello world"
    return mock_tok


class TestTextTokenizerBPE:
    def test_encode_returns_list_of_int(self):
        tok = TextTokenizer()
        tok._tokenizer = _make_mock_tokenizer()
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_ids_in_valid_range(self):
        tok = TextTokenizer()
        tok._tokenizer = _make_mock_tokenizer()
        ids = tok.encode("hello world")
        assert all(0 <= i <= 151935 for i in ids)

    def test_vocab_size_property(self):
        tok = TextTokenizer()
        tok._tokenizer = _make_mock_tokenizer()
        assert tok.vocab_size == 151936

    def test_encode_empty_string(self):
        tok = TextTokenizer()
        tok._tokenizer = _make_mock_tokenizer()
        ids = tok.encode("")
        assert isinstance(ids, list)
        assert len(ids) == 0

    def test_decode_returns_string(self):
        tok = TextTokenizer()
        tok._tokenizer = _make_mock_tokenizer()
        text = tok.decode([9906, 1917])
        assert isinstance(text, str)
