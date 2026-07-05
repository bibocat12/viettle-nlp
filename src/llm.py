import json
import re
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMEngine:
    def __init__(self, model_path: str = None, device: str = "auto"):
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model = self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        self.load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        return response.strip()

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> dict:
        response = self.generate(system_prompt, user_prompt, max_new_tokens)
        return self._parse_json(response)

    def _parse_json(self, response: str) -> dict:
        response = response.strip()
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}


_llm_instance: Optional[LLMEngine] = None


def get_llm() -> LLMEngine:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMEngine()
    return _llm_instance
