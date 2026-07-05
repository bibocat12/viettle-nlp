import re
import unicodedata


def preprocess(raw_text: str) -> dict:
    normalized = unicodedata.normalize("NFC", raw_text)
    normalized = normalized.replace("\u00a0", " ")
    sentences = split_sentences(normalized)
    return {
        "raw_text": raw_text,
        "normalized": normalized,
        "sentences": sentences,
    }


def split_sentences(text: str) -> list:
    lines = text.split("\n")
    sentences = []
    current = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current.strip():
                sentences.append(current.strip())
                current = ""
            continue
        if re.match(r"^\d+\.\s", stripped):
            if current.strip():
                sentences.append(current.strip())
            current = stripped
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if current.strip():
                sentences.append(current.strip())
            current = stripped
        elif re.match(r"^(Kết quả|Lý do|Thời điểm|Các triệu|Đặc điểm|Xử trí|Tiền sử|Bệnh sử|Đánh giá|Tình trạng)", stripped):
            if current.strip():
                sentences.append(current.strip())
            current = stripped
        else:
            if current:
                current += " " + stripped
            else:
                current = stripped
    if current.strip():
        sentences.append(current.strip())
    return sentences
