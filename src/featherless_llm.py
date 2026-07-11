"""Featherless.ai LLM — OpenAI-compatible API for Qwen2.5-7B Vietnamese Medical NER."""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
FEATHERLESS_MODEL = os.getenv("FEATHERLESS_MODEL", "PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")


class FeatherlessLLM:
    def __init__(self):
        self.api_key = FEATHERLESS_API_KEY
        self.base_url = FEATHERLESS_BASE_URL
        self.model = FEATHERLESS_MODEL
        self._available = bool(self.api_key)
        self._client = None

    def is_available(self) -> bool:
        return self._available

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
        if not self._available:
            return ""

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Featherless LLM error: {e}", file=sys.stderr)
            return ""

    def parse_json(self, response: str) -> dict:
        if not response:
            return {}
        text = response.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to extract JSON array
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # Try to extract JSON object
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}


_featherless_instance = None


def get_featherless_llm() -> FeatherlessLLM:
    global _featherless_instance
    if _featherless_instance is None:
        _featherless_instance = FeatherlessLLM()
    return _featherless_instance
