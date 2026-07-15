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

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.1,
                 tools: list = None,
                 response_format: dict = None) -> str:
        if not self._available:
            return ""

        try:
            client = self._get_client()
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if tools:
                kwargs["tools"] = tools
            if response_format:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Featherless LLM error: {e}", file=sys.stderr)
            return ""

    def generate_with_tools(self, messages: list,
                            tools: list,
                            max_tokens: int = 4096,
                            temperature: float = 0.7):
        """Send messages with tools.  Returns the full ChatCompletion message

        The caller can inspect `msg.tool_calls` to decide next step.
        """
        if not self._available:
            return None
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                tools=tools,
            )
            return resp.choices[0].message
        except Exception as e:
            print(f"Featherless LLM tool error: {e}", file=sys.stderr)
            return None

    def tool_loop(self, messages: list, tools: list,
                  tool_registry: dict,
                  max_turns: int = 8,
                  max_tokens: int = 4096,
                  temperature: float = 0.7):
        """Run a tool-calling conversation loop.

        Parameters
        ----------
        messages : list
            Conversation so far (system + user).
        tools : list
            OpenAI tool definitions.
        tool_registry : dict
            name -> callable that takes **kwargs and returns dict.
        max_turns : int
            Maximum number of tool-calling rounds.

        Returns
        -------
        str | None — final assistant content after all tool calls resolve.
        """
        for _ in range(max_turns):
            msg = self.generate_with_tools(
                messages, tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if msg is None:
                return None

            if not msg.tool_calls:
                # Normal text response — done
                return msg.content or ""

            # Execute tool calls
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = tc.function
                fn_name = fn.name
                try:
                    fn_args = json.loads(fn.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                handler = tool_registry.get(fn_name)
                if handler:
                    try:
                        result = handler(**fn_args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Ran out of turns — return last content if any
        return None

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
