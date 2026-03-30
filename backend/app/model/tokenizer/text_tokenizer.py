try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class TextTokenizer:
    _DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

    def __init__(self, model_name: str = _DEFAULT_MODEL, offline: bool = False):
        self._model_name = model_name
        self._offline = offline
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is not None:
            return
        if not _TRANSFORMERS_AVAILABLE:
            self._tokenizer = _FallbackTokenizer()
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=self._offline,
                trust_remote_code=True,
            )
        except Exception:
            self._tokenizer = _FallbackTokenizer()

    def encode(self, text: str, max_length: int = None, truncation: bool = True) -> list:
        if not text:
            return []
        self._load()
        kwargs = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = truncation
        ids = self._tokenizer.encode(text, **kwargs)
        if isinstance(ids, list):
            return ids
        return list(ids)

    def decode(self, token_ids: list) -> str:
        self._load()
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        self._load()
        return self._tokenizer.vocab_size


class _FallbackTokenizer:
    vocab_size = 151936

    def encode(self, text: str, **kwargs) -> list:
        return [ord(c) % 151936 for c in text]

    def decode(self, token_ids: list, **kwargs) -> str:
        return "".join(chr(i % 1114111 or 65) for i in token_ids)
