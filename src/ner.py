import json
import os
import re
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


NER_SYSTEM_PROMPT = """You are a medical expert. Your task is to identify and extract medical entities from the given text.
The entity types to extract are:
- SYMPTOM_AND_DISEASE: Symptoms, signs of illness, diseases, or medical conditions.
- MEDICAL_PROCEDURE: Medical procedures, surgeries, therapies, or diagnostic methods.
- DRUG: Names of medicines, drugs, vitamins, or supplements.

Return the result as a JSON list containing dictionaries with "entity" and "type" keys. If no relevant entities are found, return an empty list []"""


def _find_span_in_text(full_text: str, entity_text: str, start_hint: int = 0) -> tuple:
    idx = full_text.find(entity_text, start_hint)
    if idx >= 0:
        return idx, idx + len(entity_text)
    norm_entity = re.sub(r"\s+", " ", entity_text.strip())
    norm_full = re.sub(r"\s+", " ", full_text)
    idx = norm_full.find(norm_entity, start_hint)
    if idx >= 0:
        orig_start = 0
        norm_pos = 0
        for i, ch in enumerate(full_text):
            if norm_pos == idx:
                orig_start = i
                break
            if not ch.isspace() or (i > 0 and not full_text[i - 1].isspace()):
                norm_pos += 1
        return orig_start, orig_start + len(entity_text)
    return -1, -1


def _map_type(model_type: str, entity_text: str, context: str) -> str:
    et = entity_text.lower()
    if model_type == "DRUG":
        return "THUỐC"
    if model_type == "MEDICAL_PROCEDURE":
        test_kw = [
            "điện tâm đồ", "ecg", "x-quang", "siêu âm", "mri", "ct",
            "holter", "phân tích", "công thức", "chức năng", "troponin",
            "creatinine", "canxi", "bilirubin", "hemoglobin", "hematocrit",
            "tiểu cầu", "bạch cầu", "hồng cầu", "glucose", "protein",
            "albumin", "bảng", "xét nghiệm", "nước tiểu",
        ]
        for kw in test_kw:
            if kw in et:
                return "TÊN_XÉT_NGHIỆM"
        return "TÊN_XÉT_NGHIỆM"
    if model_type == "SYMPTOM_AND_DISEASE":
        diag_kw = [
            "viêm", "u ", "ung thư", "bệnh", "hội chứng", "suy ",
            "tăng", "giảm", "thiếu", "thừa", "đái tháo", "tăng huyết áp",
            "suy tim", "nhồi máu", "rung nhĩ", "xơ vữa", "phình",
            "ung thư", "khối u", "di căn", "loét", "nhiễm trùng",
            "viêm phổi", "viêm gan", "viêm thận", "đau thắt ngực",
            "suyễn", "hen", "lao", "viêm màng não",
        ]
        for kw in diag_kw:
            if kw in et:
                return "CHẨN_ĐOÁN"
        sent = context.lower()
        diag_ctx = ["chẩn đoán", "chẩn đoán:", "mã bệnh", "bệnh chính", "bệnh kèm"]
        for dc in diag_ctx:
            if dc in sent:
                return "CHẨN_ĐOÁN"
        return "TRIỆU_CHỨNG"
    return "TRIỆU_CHỨNG"


class QwenNER:
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
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model = self.model.eval()

    def extract(self, text: str, max_chunk: int = 1500) -> List[Dict]:
        self.load()
        chunks = self._split_chunks(text, max_chunk)
        all_mentions = []
        for chunk_text, chunk_offset in chunks:
            mentions = self._extract_chunk(chunk_text)
            for m in mentions:
                s, e = _find_span_in_text(chunk_text, m["entity"], 0)
                if s >= 0:
                    m["position"] = [s + chunk_offset, e + chunk_offset]
                    m["text"] = text[m["position"][0]:m["position"][1]]
                else:
                    m["position"] = [chunk_offset, chunk_offset + len(m["entity"])]
                    m["text"] = m["entity"]
                ctx_start = max(0, m["position"][0] - 100)
                ctx_end = min(len(text), m["position"][1] + 100)
                context = text[ctx_start:ctx_end]
                m["type"] = _map_type(m.get("type", ""), m["entity"], context)
                m["confidence"] = 0.85
                m["source"] = "qwen_ner"
            all_mentions.extend(mentions)
        return all_mentions

    def _split_chunks(self, text: str, max_chunk: int) -> list:
        if len(text) <= max_chunk:
            return [(text, 0)]
        chunks = []
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        current = ""
        current_offset = 0
        for sent in sentences:
            if len(current) + len(sent) > max_chunk and current:
                chunks.append((current, current_offset))
                current_offset += len(current)
                current = sent
            else:
                current = (current + " " + sent).strip() if current else sent
        if current.strip():
            chunks.append((current, current_offset))
        return chunks

    def _extract_chunk(self, text: str) -> List[Dict]:
        messages = [
            {"role": "system", "content": NER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {text}"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        return self._parse_response(response)

    def _parse_response(self, response: str) -> List[Dict]:
        response = response.strip()
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []
        try:
            entities = json.loads(json_match.group())
            if not isinstance(entities, list):
                return []
            result = []
            for ent in entities:
                if isinstance(ent, dict) and "entity" in ent and "type" in ent:
                    result.append({
                        "entity": str(ent["entity"]).strip(),
                        "type": str(ent["type"]).strip().upper(),
                    })
            return result
        except json.JSONDecodeError:
            return []


_qwen_ner_instance: Optional[QwenNER] = None


def get_qwen_ner() -> QwenNER:
    global _qwen_ner_instance
    if _qwen_ner_instance is None:
        _qwen_ner_instance = QwenNER()
    return _qwen_ner_instance


def extract_ner(text: str) -> List[Dict]:
    ner = get_qwen_ner()
    return ner.extract(text)
