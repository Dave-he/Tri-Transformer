import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.timeout = timeout

    def generate(self, model: str = None, prompt: str = "", **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(kwargs)
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def list_models(self) -> list:
        url = f"{self.base_url}/api/tags"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False
