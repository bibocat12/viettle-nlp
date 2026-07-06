# NER Prompt: LLM extracts keywords only, rules find positions
NER_PROMPT = """Bạn là hệ thống trích xuất thực thể y khoa tiếng Việt.
Trích xuất TẤT CẢ các keywords y tế trong văn bản. KHÔNG BỎ SÓT bất kỳ thực thể nào.

CHỈ ĐƯỢC sử dụng 5 loại thực thể sau:
- THUỐC: tên thuốc + liều (VD: "metoprolol 25mg")
- TRIỆU_CHỨNG: triệu chứng, dấu hiệu lâm sàng (VD: "đau ngực", "khó thở")
- CHẨN_ĐOÁN: chẩn đoán bệnh (VD: "tăng huyết áp", "viêm phổi")
- TÊN_XÉT_NGHIỆM: tên xét nghiệm, thủ thuật (VD: "điện tâm đồ", "chụp X-quang")
- KẾT_QUẢ_XÉT_NGHIỆM: giá trị kết quả (VD: "bình thường", "0.01")

QUAN TRỌNG:
1. Trích xuất TẤT CẢ keywords — không bỏ sót thực thể nào
2. Mỗi keyword là một khái niệm y tế ngắn gọn (2-6 từ)
3. KHÔNG trích xuất tiêu đề phần
4. KHÔNG trích xuất metadata (Vị trí:, Mức độ:)
5. Trả về JSON array: [{"entity": "...", "type": "..."}]
"""

# Prompt for LLM assertion
ASSERTION_PROMPT = """Bạn là chuyên gia xác định assertion y khoa tiếng Việt.
Xác định 3 assertion cho một mention y tế trong context.

- isNegated: Có từ phủ định trước mention? (không, chưa)
- isFamily: Liên quan đến tiền sử gia đình? (bố, mẹ, người nhà)
- isHistorical: Là thông tin quá khứ/tiền sử? (tiền sử, trước đây, đã)

Quy tắc:
1. "không X" → isNegated = true
2. "bố/mẹ/người nhà bị X" → isFamily = true
3. "tiền sử X" / "đã bị X" → isHistorical = true
4. Một mention có thể có nhiều assertion cùng lúc

Return JSON: {"isNegated": bool, "isFamily": bool, "isHistorical": bool}
"""

# Prompt for LLM type verification
TYPE_VERIFY_PROMPT = """Bạn là chuyên gia kiểm tra phân loại thực thể y khoa.
Một hệ thống rule-based đã phân loại một mention. Hãy xác nhận hoặc sửa lại.

5 loại: THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM

Quy tắc:
- THUỐC: có chứa tên thuốc + liều lượng (mg, ml,...)
- TRIỆU_CHỨNG: cảm giác bất thường của bệnh nhân (đau, khó thở, sốt,...)
- CHẨN_ĐOÁN: tên bệnh, hội chứng (viêm, ung thư, tăng huyết áp,...)
- TÊN_XÉT_NGHIỆM: xét nghiệm, thủ thuật, hình ảnh (chụp X-quang, siêu âm,...)
- KẾT_QUẢ_XÉT_NGHIỆM: kết quả xét nghiệm (bình thường, 0.01,...)

Return JSON: {"correct": true/false, "corrected_type": "...", "reason": "..."}
"""

# Prompt for LLM reranking system
LINKER_RERANK_SYSTEM = """Select best code(s) from candidates for this {mtype} mention.
Return JSON: {{"selected": ["code1", "code2"], "reason": "brief"}}"""

# Prompt for agentic linker (tool-calling loop)
LINKER_AGENT_SYSTEM = """Bạn là chuyên gia tra cứu mã bệnh (ICD-10 / RxNorm).
Bạn có các công cụ sau để tìm mã phù hợp:
{tools}

CURRENT TASK:
- Loại: {mtype}
- Mention: "{mention}"
- Context: "{context}"

Hãy suy nghĩ từng bước và gọi công cụ phù hợp.
Khi đã có kết quả, trả về JSON với "final": true.

Để gọi công cụ:
{{"thought": "suy nghĩ của bạn", "tool": "tên_công_cụ", "args": {{...}}}}

Để trả lời cuối cùng:
{{"thought": "suy nghĩ của bạn", "final": true, "candidates": ["mã1", "mã2"], "reason": "giải thích"}}"""

# Prompt for ICD-10 candidate reranking
ICD10_RERANK_PROMPT_TEMPLATE = """Select best ICD-10 code(s) from candidates for this diagnosis mention.

Mention: "{mention}"
Context: "{context}"
Type: CHẨN_ĐOÁN

Candidates:
{candidates}

Return JSON: {{"selected": ["code1", "code2"], "reason": "brief"}}"""

# Prompt for RxNorm candidate reranking
RXNORM_RERANK_PROMPT_TEMPLATE = """Select best RxNorm code(s) from candidates for this drug mention.

Mention: "{mention}"
Context: "{context}"
Type: THUỐC

Candidates:
{candidates}

Return JSON: {{"selected": ["code1", "code2"], "reason": "brief"}}"""
