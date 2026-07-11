import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

# transformers 5.x validates repo IDs strictly.
# Monkey-patch to allow local absolute paths like /kaggle/...
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None

from transformers import AutoTokenizer, AutoModel


_model = None
_tokenizer = None
_type_prototypes = {}
_device = "cpu"


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    base = os.environ.get("VIETTLE_MODELS_DIR", str(Path(__file__).parent.parent / "models"))
    model_path = os.path.join(base, "vihealthbert-base-word")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    _model = AutoModel.from_pretrained(model_path, local_files_only=True)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()


@torch.no_grad()
def _get_embedding(text: str) -> np.ndarray:
    _load_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = _model(**inputs)
    # Mean pooling over all tokens (better than CLS for sentence similarity)
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_emb = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    emb = (sum_emb / sum_mask).cpu().numpy()[0]
    return emb


def _build_prototypes():
    """Build ViHealthBERT embedding prototypes for each type."""
    global _type_prototypes
    if _type_prototypes:
        return

    type_descriptions = {
        "THUỐC": [
            "thuốc metoprolol 25mg uống",
            "aspirin 325mg", "amoxicillin 500mg",
            "tiêm insulin", "truyền dịch",
        ],
        "TRIỆU_CHỨNG": [
            "đau ngực", "khó thở", "sốt cao",
            "đau đầu", "chóng mặt", "buồn nôn",
            "mệt mỏi", "ho khan", "đau bụng",
            "căng thẳng", "lo âu", "hồi hộp",
            "đánh trống ngực", "nhịp tim nhanh",
            "mất ngủ", "chán ăn", "sụt cân",
            "tê bì", "yếu cơ", "đau khớp",
            "phù chân", "đau lưng", "khó nuốt",
            "thắt chặt ngực", "gắng sức",
            "khó chịu vùng ngực",
            "tăng đánh trống ngực",
        ],
        "CHẨN_ĐOÁN": [
            "tăng huyết áp", "viêm phổi", "đái tháo đường",
            "suy tim", "nhồi máu cơ tim", "rung nhĩ",
            "xơ gan", "ung thư phổi", "viêm gan B",
        ],
        "TÊN_XÉT_NGHIỆM": [
            "chụp X-quang ngực", "điện tâm đồ",
            "xét nghiệm máu", "siêu âm tim",
            "chụp cắt lớp vi tính",
        ],
        "KẾT_QUẢ_XÉT_NGHIỆM": [
            "kết quả bình thường", "troponin 0.01",
            "không ghi nhận bất thường", "âm tính",
            "huyết áp 120/80", "nhịp tim 80 lần/phút",
        ],
        "NON_MEDICAL": [
            "Lý do nhập viện", "Tiền sử bệnh", "Triệu chứng hiện tại",
            "Thời điểm khởi phát", "Mức độ nghiêm trọng",
            "Tình trạng ngay trước", "Kết quả xét nghiệm",
            "Kết quả hình ảnh", "Kết quả chẩn đoán hình ảnh",
            "Đặc điểm triệu chứng", "Tiếp tục cảm thấy",
            "Mất việc làm",
            "Ngày hôm nay khoảng", "Đi bộ vài chặng",
            "Không cảm nhận được triệu chứng",
            "Lên lịch tái khám", "Quyết định rằng",
            "được phục vụ tốt hơn", "ổn định và không có phàn",
            "Nhập viện khoa", "Mắt cá chân",
            "đầu là mạch", "độ bão hòa oxy",
            "ứng thất nhanh mới", "đó là nhịp tim chậm",
            "đáp ứng thất nhanh", "đặc hiệu thay đổi sóng",
            "đổi sóng t không đặc hiệu",
            "Bệnh nhân vào viện", "Theo lời bệnh nhân",
            "Một ngày người bệnh",
            # Procedures (not medical tests)
            "phẫu thuật tim", "nội soi dạ dày", "đặt stent",
            "dẫn lưu", "mổ", "shunt dẫn lưu",
        ],
    }

    for type_name, examples in type_descriptions.items():
        embs = np.array([_get_embedding(ex) for ex in examples])
        _type_prototypes[type_name] = embs.mean(axis=0)

    # Normalize prototypes
    for type_name in _type_prototypes:
        norm = np.linalg.norm(_type_prototypes[type_name])
        if norm > 0:
            _type_prototypes[type_name] /= norm


def classify_type_embedding(mention_text: str) -> Dict:
    """Classify type using ViHealthBERT semantic similarity."""
    _load_model()
    _build_prototypes()

    mention_emb = _get_embedding(mention_text)
    norm = np.linalg.norm(mention_emb)
    if norm > 0:
        mention_emb /= norm

    best_type = "TRIỆU_CHỨNG"
    best_score = -1

    for type_name, proto in _type_prototypes.items():
        score = float(np.dot(mention_emb, proto))
        if score > best_score:
            best_score = score
            best_type = type_name

    # If best match is NON_MEDICAL or score is too low, reject
    if best_type == "NON_MEDICAL" or best_score < 0.35:
        return {"type": "KHÔNG_PHẢI", "score": best_score, "source": "healthbert_embedding"}

    return {"type": best_type, "score": best_score, "source": "healthbert_embedding"}


# Keyword-based fallback patterns
TYPE_KEYWORDS = {
    "THUỐC": [
        r'\d+\s*(?:mg|ml|mcg|IU|U)\b',
        r'\b(?:po|iv|im|sc|sl|tiêm|đặt|nhỏ|bôi|thoa)\b',
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
    """Classify mention type — keyword first (drugs/tests/results), then embedding."""
    import re
    text_lower = mention_text.lower()

    # 0. Reject non-entity patterns (instructions, fragments, verbs)
    NON_ENTITY_PATTERNS = [
        'uống hôm nay', 'uống hàng chục', 'được chỉ định',
        'bắt đầu dùng', 'cho thấy', 'liên quan đến',
        'là không', 'đã sử dụng',
        'vùng', 'sau đó', 'sau khi', 'ngày hôm nay',
        'đến khoa', 'khoa điều trị', 'được thăm khám',
        'gì đáng chú ý', 'nhận gì bất thường',
        'định', 'được cho dùng', 'tiếp tục cảm thấy',
        'còn cảm giác',
    ]
    has_drug = any(re.search(kw, text_lower) for kw in TYPE_KEYWORDS["THUỐC"])
    has_test = any(kw in text_lower for kw in TYPE_KEYWORDS["TÊN_XÉT_NGHIỆM"] if isinstance(kw, str))
    has_result = any(kw in text_lower for kw in TYPE_KEYWORDS["KẾT_QUẢ_XÉT_NGHIỆM"] if isinstance(kw, str))

    # Check for non-entity verbs/instructions
    verb_count = sum(1 for p in NON_ENTITY_PATTERNS if p in text_lower)

    # If the phrase has non-entity patterns AND no real medical keyword, reject
    if verb_count >= 1 and not (has_drug or has_test or has_result):
        is_known_symptom = any(kw in text_lower for kw in ['đau', 'khó thở', 'sốt', 'buồn nôn', 'mệt mỏi', 'ho'])
        if not is_known_symptom:
            return {"type": "KHÔNG_PHẢI", "score": 0.90, "source": "non_entity_filter"}

    # If phrase has non-entity verbs PLUS medical content, only allow if
    # the medical content is the MAIN part (not followed by verbs)
    if verb_count >= 1 and (has_drug or has_test or has_result):
        words = text_lower.split()
        non_entity_words = {'cho', 'thấy', 'là', 'không', 'được', 'chỉ', 'định',
                           'sau', 'đó', 'khi', 'của', 'và', 'hoặc', 'các'}
        for i, w in enumerate(words):
            if w in non_entity_words and i > 0 and i < len(words) - 1:
                return {"type": "KHÔNG_PHẢI", "score": 0.88, "source": "trailing_verb"}

    # Also check trailing verbs for drug phrases even without verb_count match
    # e.g. "doxycycline cho viêm tuyến mồ hôi" has "cho" as trailing non-drug word
    if has_drug and not has_test and not has_result:
        words = text_lower.split()
        # If drug name is followed by non-drug content, reject
        drug_keywords_found = [kw for kw in TYPE_KEYWORDS["THUỐC"] if re.search(kw, text_lower)]
        if drug_keywords_found:
            first_drug_pos = len(text_lower)
            for kw in drug_keywords_found:
                m = re.search(kw, text_lower)
                if m and m.start() < first_drug_pos:
                    first_drug_pos = m.start()
            # If drug name starts early but phrase is much longer, reject
            if first_drug_pos == 0 and len(text_lower) > 30:
                return {"type": "KHÔNG_PHẢI", "score": 0.88, "source": "drug_phrase_too_long"}

    # 0b. If test name + result combined (e.g. "ecg bình thường"), reject

    # 0b. If test name + result combined (e.g. "ecg bình thường"), reject
    if has_test and has_result:
        return {"type": "KHÔNG_PHẢI", "score": 0.90, "source": "test_result_combined"}

    # 0. Reject procedures/surgeries (not medical tests)
    PROCEDURE_KW = ['phẫu thuật', 'mổ', 'nội soi', 'đặt stent', 'dẫn lưu',
                    'shunt', 'thông tim', 'chụp mạch', 'đặt ống']
    for kw in PROCEDURE_KW:
        if kw in text_lower:
            return {"type": "KHÔNG_PHẢI", "score": 0.95, "source": "procedure_filter"}

    # 1. Strong keyword matches (high precision, never filtered)
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

    # 2. ViHealthBERT embedding (including non-medical rejection)
    result = None
    try:
        result = classify_type_embedding(mention_text)
        if result["type"] == "KHÔNG_PHẢI":
            return result
    except Exception as e:
        print(f"ViHealthBERT embedding error: {e}", file=__import__('sys').stderr)

    # 3. Diagnosis keywords (lower precision, use after embedding non-medical filter)
    for kw in TYPE_KEYWORDS["CHẨN_ĐOÁN"]:
        if kw in text_lower:
            return {"type": "CHẨN_ĐOÁN", "score": 0.75, "source": "keyword"}

    # 4. Fall back to embedding result
    try:
        if result and result["score"] > 0.3:
            return result
    except Exception:
        pass

    # 5. Default
    return {"type": "TRIỆU_CHỨNG", "score": 0.60, "source": "default"}


def classify_assertion(mention_text: str, context: str) -> Dict:
    """Classify assertion using keyword patterns."""
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
