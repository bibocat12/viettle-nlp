import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

RUNPOD_API_URL = "https://api.runpod.ai/v2/c4qbicuam59iw1"
RUNPOD_API_KEY = os.getenv("RUNPOD_APIKEY", "")

SYS_START = chr(10022) + "im_start" + chr(10022) + "system\n"
IM_END = chr(10022) + "im_end" + chr(10022) + "\n"
USER_START = chr(10022) + "im_start" + chr(10022) + "user\n"
ASSIST_START = chr(10022) + "im_start" + chr(10022) + "assistant\n"


class RunPodLLM:
    def __init__(self):
        self.api_url = RUNPOD_API_URL
        self.api_key = RUNPOD_API_KEY
        self._available = bool(self.api_key)

    def is_available(self) -> bool:
        return self._available

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        if not self._available:
            return ""

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

        prompt = SYS_START + system_prompt + IM_END + USER_START + user_prompt + IM_END + ASSIST_START

        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            resp = requests.post(
                self.api_url + "/run",
                json=payload,
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()
            job_data = resp.json()
            job_id = job_data.get("id", "")

            if not job_id:
                return ""

            for _ in range(60):
                time.sleep(2)
                status_resp = requests.get(
                    self.api_url + "/status/" + job_id,
                    headers=headers,
                    timeout=10
                )
                status_data = status_resp.json()
                status = status_data.get("status", "")

                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    raw = ""
                    if isinstance(output, dict):
                        raw = output.get("text", "")
                    elif isinstance(output, str):
                        raw = output
                    else:
                        raw = str(output)
                    return self._extract_text(raw)
                elif status in ("FAILED", "CANCELLED"):
                    return ""

            return ""

        except Exception as e:
            print("RunPod LLM error: " + str(e), file=sys.stderr)
            return ""

    def _extract_text(self, raw: str) -> str:
        text = raw.strip()
        # Try to parse as RunPod response format
        # Format: [{'choices': [{'tokens': ['text']}]}]
        # or: [{"choices": [{"tokens": ["text"]}]}]
        
        # First try standard JSON
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0:
                choices = data[0].get("choices", [])
                if choices:
                    tokens = choices[0].get("tokens", [])
                    if tokens:
                        result = "".join(tokens)
                        return self._clean_response(result)
        except json.JSONDecodeError:
            pass
        
        # Try with single quotes replaced (Python-style)
        try:
            json_text = text.replace("'", '"')
            data = json.loads(json_text)
            if isinstance(data, list) and len(data) > 0:
                choices = data[0].get("choices", [])
                if choices:
                    tokens = choices[0].get("tokens", [])
                    if tokens:
                        result = "".join(tokens)
                        return self._clean_response(result)
        except json.JSONDecodeError:
            pass
        
        # Try regex extraction
        m = re.search(r"'tokens':\s*\['(.*?)'\]", text, re.DOTALL)
        if m:
            return self._clean_response(m.group(1))
        
        # Try to find JSON array directly
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            return self._clean_response(m.group())
        
        return self._clean_response(text)

    def _clean_response(self, text: str) -> str:
        """Clean up response text by removing noise."""
        # Remove common noise patterns
        text = text.replace("✦", "").replace("\\n", "")
        text = re.sub(r'Jimmy\s*(End|Start)', '', text)
        text = text.strip()
        # Find the last ] or } and truncate
        for i in range(len(text) - 1, -1, -1):
            if text[i] in (']', '}'):
                return text[:i + 1]
        return text

    def parse_json(self, response: str) -> dict:
        # First extract text from RunPod format
        text = self._extract_text(response)
        
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try with single quotes replaced
        try:
            return json.loads(text.replace("'", '"'))
        except:
            pass
        
        # Try to find JSON array
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
        
        # Try to find JSON object
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
        
        return {}

    def extract_text_from_response(self, response: str) -> str:
        """Extract text from RunPod response format: [{'choices': [{'tokens': ['text']}]}]"""
        try:
            data = json.loads(response)
            if isinstance(data, list) and len(data) > 0:
                choices = data[0].get("choices", [])
                if choices:
                    tokens = choices[0].get("tokens", [])
                    if tokens:
                        return "".join(tokens)
        except:
            pass
        return response

    def batch_generate(self, prompts: list, max_tokens: int = 512, max_workers: int = 4) -> list:
        """Generate multiple prompts in parallel.

        Args:
            prompts: list of (system_prompt, user_prompt) tuples
            max_tokens: max tokens per generation
            max_workers: max parallel workers

        Returns:
            list of response strings (same order as input)
        """
        if not self._available:
            return [""] * len(prompts)

        results = [None] * len(prompts)

        def _generate_one(idx_prompts):
            idx, (system_prompt, user_prompt) = idx_prompts
            response = self.generate(system_prompt, user_prompt, max_tokens)
            return idx, response

        with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as executor:
            futures = [executor.submit(_generate_one, (i, p)) for i, p in enumerate(prompts)]
            for future in as_completed(futures):
                try:
                    idx, response = future.result()
                    results[idx] = response
                except Exception as e:
                    print("RunPod batch error: " + str(e), file=sys.stderr)

        return [r if r is not None else "" for r in results]


_runpod_llm_instance = None


def get_runpod_llm() -> RunPodLLM:
    global _runpod_llm_instance
    if _runpod_llm_instance is None:
        _runpod_llm_instance = RunPodLLM()
    return _runpod_llm_instance
