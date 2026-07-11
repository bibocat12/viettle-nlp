"""Summarizer agent: lists attributes found in each paragraph section."""

import sys, json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents import Agent, Tool, AgentContext
from agents.section_parser import get_section_summary

# Mapping of section titles to expected attributes
SECTION_ATTR_MAP = {
    "tiền sử bệnh": ["history_medications", "family_history", "risk_factors", "social_history"],
    "tiền sử bệnh hiện tại": ["reason_for_admission", "symptom_onset", "current_symptoms", "symptom_progression"],
    "đánh giá tại bệnh viện": ["vital_signs", "physical_exam", "test_results", "imaging"],
    "bệnh sử": ["history_medications", "current_symptoms", "past_medical_history"],
    "kết quả xét nghiệm": ["lab_values", "normal_findings", "abnormal_findings"],
    "kết quả chẩn đoán hình ảnh": ["imaging_findings", "impression"],
}


class SummarizerAgent(Agent):
    """Lists attributes found in each paragraph/section."""
    def __init__(self, llm=None):
        tools = [
            Tool("summarize_section", "Summarize a section's attributes", self._summarize),
        ]
        super().__init__("summarizer", llm, tools)

    def _summarize(self, text: str, section_title: str) -> dict:
        # Determine expected attributes
        title_lower = section_title.lower().strip()
        expected_attrs = []
        for key, attrs in SECTION_ATTR_MAP.items():
            if key in title_lower:
                expected_attrs = attrs
                break
        if not expected_attrs:
            # Guess from content
            if any(w in text.lower() for w in ["thuốc", "mg", "bid", "tid"]):
                expected_attrs.append("medications")
            if any(w in text.lower() for w in ["triệu chứng", "đau", "khó thở", "sốt"]):
                expected_attrs.append("symptoms")
            if any(w in text.lower() for w in ["chẩn đoán", "bệnh", "hội chứng"]):
                expected_attrs.append("diagnoses")
            if any(w in text.lower() for w in ["xét nghiệm", "kết quả", "ecg", "x-quang"]):
                expected_attrs.append("test_results")

        # Use LLM for smarter summarization
        if self.llm:
            prompt = f"""Phân tích đoạn văn bản y khoa sau và liệt kê các thông tin có trong đó.

Tiêu đề: "{section_title}"
Đoạn văn: "{text[:500]}"

Các thông tin cần tìm:
{', '.join(expected_attrs) if expected_attrs else 'thuốc, triệu chứng, chẩn đoán, xét nghiệm, kết quả'}

Trả về JSON: {{"attributes": ["thuốc: metoprolol", "triệu chứng: đau ngực"], "summary": "câu tóm tắt ngắn"}}"""
            resp = self.llm.generate(
                "Bạn là chuyên gia tóm tắt hồ sơ bệnh án.",
                prompt, 300
            )
            try:
                return json.loads(resp)
            except:
                pass

        # Rule-based fallback
        found = []
        if any(kw in text.lower() for kw in ["metoprolol", "aspirin", "mg", "thuốc"]):
            found.append("thuốc")
        if any(kw in text.lower() for kw in ["đau", "khó thở", "sốt", "buồn nôn"]):
            found.append("triệu chứng")
        if any(kw in text.lower() for kw in ["chẩn đoán", "bệnh", "viêm", "tăng huyết áp"]):
            found.append("chẩn đoán")
        if any(kw in text.lower() for kw in ["x-quang", "ecg", "siêu âm", "xét nghiệm"]):
            found.append("xét nghiệm")
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return {
            "attributes": found,
            "summary": (lines[0] if lines else "")[:150],
        }

    def run(self, ctx: AgentContext, sections: list) -> list:
        """Summarize each section. Returns list of summaries."""
        self.log("start", {"sections": len(sections)})
        summaries = []
        for sec in sections:
            summary = self.call_tool("summarize_section",
                text=sec["text"], section_title=sec["title"])
            summaries.append({
                "section_id": sec["id"],
                "title": sec["title"],
                "attributes": summary.get("attributes", []),
                "section_summary": summary.get("summary", ""),
            })
        self.log("complete", {"summaries": len(summaries)})
        return summaries
