"""OpenAI-compatible vLLM client used by the competition pipeline."""

import json
import os
import re
import sys
import time
from typing import Any, Optional

import requests


DEFAULT_MODEL = "bibocat/qwen3-ner-grpo-v2-merged"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMLLM:
    """Small adapter exposing the same interface as the other LLM backends."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL") or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.getenv("VLLM_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
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

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate one completion, optionally constrained by vLLM JSON Schema.

        ``response_format`` is deliberately passed through unchanged.  This keeps
        the adapter compatible with the OpenAI-compatible vLLM endpoint while
        allowing callers that do not need structured output to keep using the
        lightweight text interface.
        """
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
        if response_format is not None:
            payload["response_format"] = response_format
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()
                if not content:
                    raise ValueError("vLLM returned an empty completion")
                self.last_error = ""
                return content
            except Exception as exc:
                self.last_error = str(exc)
                if attempt < self.max_retries:
                    # Back off 1s, then 2s by default.  A failed chunk is
                    # isolated by the fusion runner, so do not retry forever.
                    time.sleep(float(attempt))
                    continue
                print(
                    f"vLLM request failed after {self.max_retries} attempts: {exc}",
                    file=sys.stderr,
                )
        raise RuntimeError(
            f"vLLM generation failed after {self.max_retries} attempts: {self.last_error}"
        )

    TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Send a chat request with optional tool definitions.

        Returns the full response *message* dict (with ``content`` and
        optionally ``tool_calls``).  The caller is responsible for executing
        any tool calls and continuing the conversation loop.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "tool_choice": "auto",
        }
        if tools is not None:
            payload["tools"] = tools
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                # Some vLLM versions do not parse tool calls from Qwen's
                # <tool_call> blocks.  Fall back to manual extraction.
                if tools and not message.get("tool_calls"):
                    parsed = self._parse_tool_calls(message.get("content", ""))
                    if parsed:
                        message["tool_calls"] = parsed
                        message["content"] = None
                self.last_error = ""
                return message
            except Exception as exc:
                self.last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(float(attempt))
                    continue
                print(
                    f"vLLM tool-call request failed after {self.max_retries} attempts: {exc}",
                    file=sys.stderr,
                )
        raise RuntimeError(
            f"vLLM tool-call generation failed after {self.max_retries} attempts: {self.last_error}"
        )

    def _parse_tool_calls(self, content: str) -> Optional[list[dict]]:
        """Extract tool calls from Qwen <tool_call> blocks in content."""
        if not content:
            return None
        calls = []
        for match in self.TOOL_CALL_RE.finditer(content):
            try:
                obj = json.loads(match.group(1))
                name = obj.get("name", "")
                arguments = obj.get("arguments", {})
                calls.append({
                    "id": f"call_{len(calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                })
            except (json.JSONDecodeError, TypeError):
                continue
        return calls or None

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

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        schema: Optional[dict] = None,
    ) -> Any:
        response_format = None
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "medical_entity_proposals",
                    "strict": True,
                    "schema": schema,
                },
            }
        return self.parse_json(
            self.generate(system_prompt, user_prompt, max_tokens, response_format=response_format)
        )


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
