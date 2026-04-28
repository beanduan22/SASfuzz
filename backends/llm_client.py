from __future__ import annotations

import json
import logging
import time
from typing import Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

@runtime_checkable
class LLMBackend(Protocol):

    @property
    def current_model(self) -> str:
        ...

    def generate(self, prompt: str, advance: bool = True) -> str:
        ...

    def stats(self) -> dict:
        ...

OLLAMA_URL = "http://localhost:11434"

_DEFAULT_OLLAMA_MODELS = [
    "qwen2.5-coder:32b",
]

class OllamaClient:

    def __init__(
        self,
        models: list[str] = _DEFAULT_OLLAMA_MODELS,
        base_url: str = OLLAMA_URL,
        timeout: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> None:
        self._models = models
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._idx = 0
        self._call_counts = {m: 0 for m in models}

    @property
    def current_model(self) -> str:
        return self._models[self._idx % len(self._models)]

    def generate(self, prompt: str, advance: bool = True) -> str:
        model = self.current_model
        if advance:
            self._idx += 1

        logger.info("LLM call → %s", model)
        self._call_counts[model] += 1

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p,
                "num_predict": self._max_tokens,
            },
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timeout after {self._timeout}s for model {model}"
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        elapsed = time.time() - start
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON: {resp.text[:200]}"
            ) from exc

        response_text = data.get("response", "")
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(response_text))
        return response_text

    def stats(self) -> dict:
        return {"call_counts": dict(self._call_counts), "next_model": self.current_model}

class OpenAIClient:

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install openai") from None
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_count = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    @property
    def current_model(self) -> str:
        return self._model

    def generate(self, prompt: str, advance: bool = True) -> str:
        self._call_count += 1
        logger.info("LLM call → %s", self._model)
        start = time.time()
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        elapsed = time.time() - start
        text = resp.choices[0].message.content or ""
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(text))
        return text

    def stats(self) -> dict:
        return {"call_counts": {self._model: self._call_count}, "next_model": self._model}

class AnthropicClient:

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic") from None
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_count = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def current_model(self) -> str:
        return self._model

    def generate(self, prompt: str, advance: bool = True) -> str:
        self._call_count += 1
        logger.info("LLM call → %s", self._model)
        start = time.time()
        msg = self._get_client().messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - start
        text = msg.content[0].text if msg.content else ""
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(text))
        return text

    def stats(self) -> dict:
        return {"call_counts": {self._model: self._call_count}, "next_model": self._model}

class DeepSeekClient:
    """DeepSeek API (OpenAI-compatible at api.deepseek.com)."""
    def __init__(self, model="deepseek-chat", temperature=0.7, max_tokens=4096):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install openai") from None
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_count = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            api_key = os.environ.get("DEEPSEEK_API_KEY", "placeholder")
            self._client = openai.OpenAI(
                base_url="https://api.deepseek.com/v1", api_key=api_key
            )
        return self._client

    @property
    def current_model(self): return self._model

    def generate(self, prompt, advance=True):
        self._call_count += 1
        logger.info("LLM call → %s", self._model)
        start = time.time()
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        elapsed = time.time() - start
        text = resp.choices[0].message.content or ""
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(text))
        return text

    def stats(self): return {"call_counts": {self._model: self._call_count}, "next_model": self._model}

_BACKEND_MAP = {
    "deepseek-v2": (DeepSeekClient,  "deepseek-chat"),
    "gpt5":        (OpenAIClient,    "gpt-5"),
    "claude35":    (AnthropicClient, "claude-3-5-sonnet-20241022"),
}

def create_client(backend: str = "deepseek-v2", model: str | None = None) -> LLMBackend:
    if backend not in _BACKEND_MAP:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_BACKEND_MAP)}")
    cls, default = _BACKEND_MAP[backend]
    if cls is OpenAIClient:
        return OpenAIClient(model=model or default)
    elif cls is AnthropicClient:
        return AnthropicClient(model=model or default)
    elif cls is DeepSeekClient:
        return DeepSeekClient(model=model or default)
    raise ValueError(f"Unexpected class for backend '{backend}'")
