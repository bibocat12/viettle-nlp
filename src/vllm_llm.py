"""OpenAI-compatible vLLM client used by the competition pipeline."""

import json
import os
import sys
from typing import Any, Optional

import requests


DEFAULT_MODEL = "bibocat/qwen3-ner-grpo-final"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMLLM:
    """Small adapter exposing the same interface as the other LLM backends."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL") or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.getenv("VLLM_API_KEY", "")
        self.timeout = timeout
        self.last_error = ""

    @property
    def model_name(self) -> str:
        return self.model

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_available(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=min(self.timeout, 8),
            )
            response.raise_for_status()
            payload = response.json()
            served = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
            if self.model not in served:
                self.last_error = f"model {self.model!r} is not served; available={sorted(served)}"
                return False
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.last_error = ""
            return (data["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"vLLM request failed: {exc}", file=sys.stderr)
            return ""

    @staticmethod
    def parse_json(response: str) -> Any:
        """Decode the first JSON array/object, tolerating Qwen's `.JSON` prefix."""
        if not response:
            return {}
        decoder = json.JSONDecoder()
        for index, char in enumerate(response):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(response[index:])
                return value
            except json.JSONDecodeError:
                continue
        return {}

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> Any:
        return self.parse_json(self.generate(system_prompt, user_prompt, max_tokens))


_vllm_instance: Optional[VLLMLLM] = None


def get_vllm_llm(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> VLLMLLM:
    global _vllm_instance
    requested_url = (base_url or os.getenv("VLLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    requested_model = model or os.getenv("VLLM_MODEL") or DEFAULT_MODEL
    if (
        _vllm_instance is None
        or _vllm_instance.base_url != requested_url
        or _vllm_instance.model != requested_model
    ):
        _vllm_instance = VLLMLLM(base_url=requested_url, model=requested_model)
    return _vllm_instance
