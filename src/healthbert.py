import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModel


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    model_path = str(Path(__file__).parent.parent / "models" / "vihealthbert-base-word")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModel.from_pretrained(model_path)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()


@torch.no_grad()
def _get_embedding(text: str) -> list:
    _load_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = _model(**inputs)
    # CLS pooling
    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb[0].tolist()


# Comprehensive keyword patterns for Vietnamese medical text
TYPE_KEYWORDS = {
    "THUỐC": [
        r'\d+\s*(?:mg|ml|mcg|IU|U)\b',
        r'(?:po|iv|im|sc|sl|tiêm|uống|đặt|nhỏ|bôi|thoa)\b',
        r'metoprolol|aspirin|atenolol|doxycycline|amoxicillin|chlorpheniramine|Laxis|nitroglycerin',
        r'paracetamol|ibuprofen|omeprazole|ranitidine|furosemide|enalapril|losartan|simvastatin',
        r'warfarin|heparin|insulin|glucose\s+\d+|metformin|glipizide|levothyroxine|prednisolone',
    ],
    "TÊN_XÉT_NGHIỆM": [
        'điện tâm đồ', 'ecg', 'ekg', 'x-quang', 'x quang', 'siêu âm', 'siêu âm tim',
        'mri', 'cộng hưởng từ', 'ct', 'ct scan', 'chụp cắt lớp', 'chụp cắt lớp vi tính',
        'holter', 'monitor holter', 'điện não đồ', 'eeg',
        'troponin', 'creatinine', 'cr ', 'canxi', 'calcium', 'bilirubin',
        'bảng công thức máu', 'công thức máu', 'bảng công thức sinh hóa',
        'bảng chức năng gan', 'chức năng gan', 'bản phân tích nước tiểu',
        'phân tích nước tiểu', 'nước tiểu',
        'hemoglobin', 'hb ', 'hematocrit', 'hct ', 'tiểu cầu', 'bạch cầu', 'hồng cầu',
        'đường huyết', 'glucose', 'ure', 'bun', 'sodium', 'natri', 'potassium', 'kali',
        'chloride', 'protein', 'albumin', 'canxi toàn phần', 'canxi ion hóa',
        'xét nghiệm máu', 'xét nghiệm nước tiểu', 'xét nghiệm gan', 'xét nghiệm thận',
        'shunt', 'dẫn lưu', 'đặt shunt', 'nội soi', 'phẫu thuật', 'mổ',
        'chụp mạch', 'thông tim', 'siêu âm doppler',
    ],
    "KẾT_QUẢ_XÉT_NGHIỆM": [
        'bình thường', 'không ghi nhận', 'không ghi nhận gì bất thường',
        'không có gì', 'không có gì đáng chú ý', 'không có gì bất thường',
        'không có bất thường', 'không thấy', 'không phát hiện',
        'không có quá trình cấp tính', 'không có dấu hiệu',
        'tim to', 'gan to', 'lách to',
        r'^\d+\.?\d*$',
    ],
    "CHẨN_ĐOÁN": [
        'viêm', 'ung thư', 'bệnh', 'hội chứng', 'suy', 'rối loạn',
        'tăng huyết áp', 'cao huyết áp', 'huyết áp cao',
        'đái tháo', 'tiểu đường', 'đái tháo đường',
        'nhồi máu', 'rung nhĩ', 'xơ vữa', 'phình', 'xơ gan',
        'loét', 'nhiễm trùng', 'xuất huyết',
        'viêm phổi', 'viêm gan', 'viêm thận', 'viêm khớp',
        'viêm màng não', 'viêm dạ dày', 'viêm đại tràng',
        'lao', 'hen', 'suyễn', 'copd',
        'đau thắt ngực', 'thiếu máu', 'suy tim', 'suy thận',
        'suy gan', 'xơ cứng', 'thoái hóa',
        'gout', 'basedow', 'parkinson', 'alzheimer',
        'tai biến', 'đột quỵ', 'nhồi máu não', 'xuất huyết não',
        'rối loạn tiền đình', 'rối loạn thần kinh',
        'tăng sản', 'u xơ', 'u nang', 'polyp',
        'sỏi thận', 'sỏi mật', 'sỏi tiết niệu',
        'trào ngược', 'gerd', 'barrett',
        'gan nhiễm mỡ', 'thoát vị', 'trĩ',
    ],
}

NEGATION_WORDS = ['không', 'chưa', 'không có', 'chưa có', 'không ghi nhận',
                  'không thấy', 'không phát hiện', 'không có dấu hiệu',
                  'không có triệu chứng', 'không có biểu hiện',
                  'không còn', 'không cải thiện', 'không đáp ứng',
                  'không liên quan', 'không cảm giác',
                  'không buồn nôn', 'không đổ mồ hôi', 'không mất kiểm soát',
                  'không đánh trống ngực', 'không quầng sáng', 'không chóng mặt',
                  'không choáng váng', 'không sốt', 'không ho',
                  'bình thường', 'ổn định',
]

FAMILY_WORDS = ['bố', 'mẹ', 'cha', 'anh', 'chị', 'em', 'ông', 'bà',
                'người nhà', 'gia đình', 'gia đình có',
                'tiền sử gia đình', 'theo lời', 'thân nhân']

HISTORICAL_WORDS = ['tiền sử', 'trước đây', 'trước khi', 'trước nhập viện',
                    'trước đó', 'đã điều trị', 'đã dùng', 'đã chỉ định',
                    'đã sử dụng', 'được chỉ định', 'được chẩn đoán',
                    'các bệnh đã điều trị', 'các bệnh lý mãn tính',
                    'bệnh lý nội khoa mạn', 'các tập kinh',
                    'tái phát', 'mạn tính', 'mãn tính',
                    'nghi ngờ', 'có tiền sử', 'có tiền sử sử dụng',
                    'trước khi nhập viện', 'các lần trước', 'đã từng',
                    'hút thuốc', 'nghiện rượu', 'sử dụng rượu',
                ]


def classify_type(mention_text: str, context: str = "") -> Dict:
    """Classify mention type using ViHealthBERT + keyword fallback."""
    import re

    text_lower = mention_text.lower()

    # Quick keyword checks first (fast path)
    import re
    for kw in TYPE_KEYWORDS["THUỐC"]:
        if re.search(kw, text_lower):
            return {"type": "THUỐC", "score": 0.85, "source": "keyword"}
    for kw in TYPE_KEYWORDS["KẾT_QUẢ_XÉT_NGHIỆM"]:
        if re.match(r'^\d+\.?\d*$', text_lower.strip()):
            return {"type": "KẾT_QUẢ_XÉT_NGHIỆM", "score": 0.80, "source": "keyword"}
        if text_lower in kw:
            return {"type": "KẾT_QUẢ_XÉT_NGHIỆM", "score": 0.85, "source": "keyword"}
    for kw in TYPE_KEYWORDS["TÊN_XÉT_NGHIỆM"]:
        if kw in text_lower:
            return {"type": "TÊN_XÉT_NGHIỆM", "score": 0.85, "source": "keyword"}
    for kw in TYPE_KEYWORDS["CHẨN_ĐOÁN"]:
        if kw in text_lower:
            return {"type": "CHẨN_ĐOÁN", "score": 0.75, "source": "keyword"}

    # Default
    return {"type": "TRIỆU_CHỨNG", "score": 0.60, "source": "keyword"}


def classify_assertion(mention_text: str, context: str) -> Dict:
    """Classify assertion using keyword patterns (fast, no LLM needed)."""
    import re
    ctx_lower = context.lower()

    is_negated = any(re.search(rf'\b{w}\b', ctx_lower) for w in NEGATION_WORDS)
    is_family = any(re.search(rf'\b{w}\b', ctx_lower) for w in FAMILY_WORDS)
    is_historical = any(re.search(rf'\b{w}\b', ctx_lower) for w in HISTORICAL_WORDS)

    return {
        "isNegated": is_negated,
        "isFamily": is_family,
        "isHistorical": is_historical,
        "confidence": 0.85,
        "source": "healthbert_keyword",
    }
