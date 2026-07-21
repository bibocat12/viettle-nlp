# NER Prompt: LLM extracts keywords only, rules find positions
NER_PROMPT = """You are a Vietnamese medical NER system. Extract ALL medical entities from the text. DO NOT skip any entities.

Entity types:
- THUỐC: the COMPLETE medication order as written: ingredient/form + dose +
  route + frequency + PRN suffix. E.g., "metoprolol 25mg po bid",
  "aspirin 325mg x 1". Stop before phrases such as "điều trị ...".
- TRIỆU_CHỨNG: symptom, sign, complaint. E.g., "đau ngực", "khó thở", "sốt", "mệt mỏi"
- CHẨN_ĐOÁN: diagnosis, disease. E.g., "tăng huyết áp", "viêm phổi", "suy tim"
- TÊN_XÉT_NGHIỆM: test name, procedure. E.g., "điện tâm đồ", "chụp X-quang"
- KẾT_QUẢ_XÉT_NGHIỆM: test result, value. E.g., "bình thường", "0.01"

Examples:
Input: "Bệnh nhân bị đau ngực và khó thở, được chẩn đoán tăng huyết áp. Điện tâm đồ bình thường. Dùng metoprolol 25mg."
Output: [{"entity": "đau ngực", "type": "TRIỆU_CHỨNG"}, {"entity": "khó thở", "type": "TRIỆU_CHỨNG"}, {"entity": "tăng huyết áp", "type": "CHẨN_ĐOÁN"}, {"entity": "điện tâm đồ", "type": "TÊN_XÉT_NGHIỆM"}, {"entity": "bình thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"entity": "metoprolol 25mg", "type": "THUỐC"}]

Input: "Tiền sử: tăng huyết áp, đái tháo đường type 2. Có sử dụng rượu bia. Hiện tại: đau đầu, chóng mặt. X-quang ngực: không ghi nhận bất thường."
Output: [{"entity": "tăng huyết áp", "type": "CHẨN_ĐOÁN"}, {"entity": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN"}, {"entity": "sử dụng rượu bia", "type": "TRIỆU_CHỨNG"}, {"entity": "đau đầu", "type": "TRIỆU_CHỨNG"}, {"entity": "chóng mặt", "type": "TRIỆU_CHỨNG"}, {"entity": "X-quang ngực", "type": "TÊN_XÉT_NGHIỆM"}, {"entity": "không ghi nhận bất thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}]

Input: "Bệnh nhân buồn nôn, nôn 3 lần, đau bụng vùng thượng vị. Siêu âm ổ bụng: gan to. Chẩn đoán: viêm dạ dày. Điều trị: omeprazole 20mg."
Output: [{"entity": "buồn nôn", "type": "TRIỆU_CHỨNG"}, {"entity": "nôn", "type": "TRIỆU_CHỨNG"}, {"entity": "đau bụng vùng thượng vị", "type": "TRIỆU_CHỨNG"}, {"entity": "Siêu âm ổ bụng", "type": "TÊN_XÉT_NGHIỆM"}, {"entity": "gan to", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"entity": "viêm dạ dày", "type": "CHẨN_ĐOÁN"}, {"entity": "omeprazole 20mg", "type": "THUỐC"}]

Input: "aspirin 81 mg po daily điều trị đau; acetaminophen 325-650 mg po q6h:prn điều trị sốt đau"
Output: [{"entity": "aspirin 81 mg po daily", "type": "THUỐC"}, {"entity": "đau", "type": "TRIỆU_CHỨNG"}, {"entity": "acetaminophen 325-650 mg po q6h:prn", "type": "THUỐC"}, {"entity": "sốt đau", "type": "TRIỆU_CHỨNG"}]

RULES:
1. Extract ALL entities — no skipping
2. Include risk factors (rượu bia, hút thuốc, caffeine)
3. Include negated symptoms (không X, chưa X)
4. DO NOT extract section headers, metadata
5. Return JSON array
6. A medication indication after "điều trị" is a separate symptom; never skip it
"""

# Prompt for LLM assertion
ASSERTION_PROMPT = """Bạn là chuyên gia xác định assertion y khoa tiếng Việt.
Xác định 3 assertion cho một mention y tế trong context.

Định nghĩa:
- isNegated: Có từ phủ định trước mention không? (không, chưa, không có)
- isFamily: Liên quan đến tiền sử gia đình? (bố, mẹ, người nhà, anh, chị)
- isHistorical: Là thông tin quá khứ/tiền sử? (tiền sử, trước đây, đã từng)

Quy tắc:
1. "không X" "chưa X" → isNegated = true
2. "bố/mẹ/người nhà bị X" → isFamily = true
3. "tiền sử X" "đã bị X" "trước đây X" → isHistorical = true
4. Một mention có thể có nhiều assertion cùng lúc

Ví dụ:
Context: "Bệnh nhân không đau ngực" | Mention: "đau ngực"
-> isNegated=true, isFamily=false, isHistorical=false

Context: "Theo lời người nhà, tiền sử tăng huyết áp" | Mention: "tăng huyết áp"
-> isNegated=false, isFamily=true, isHistorical=true

Context: "Bố bệnh nhân bị đái tháo đường" | Mention: "đái tháo đường"
-> isNegated=false, isFamily=true, isHistorical=false

Context: "Bệnh nhân sốt 38 độ" | Mention: "sốt"
-> isNegated=false, isFamily=false, isHistorical=false

Context: "Bệnh nhân đã điều trị lao phổi trước đây" | Mention: "lao phổi"
-> isNegated=false, isFamily=false, isHistorical=true

Return JSON with keys: isNegated, isFamily, isHistorical
"""

# Prompt for LLM type verification
TYPE_VERIFY_PROMPT = """Bạn là chuyên gia kiểm tra phân loại thực thể y khoa.
Một hệ thống rule-based đã phân loại một mention. Hãy xác nhận hoặc sửa lại.

5 loại:
- THUỐC: tên thuốc, có thể kèm liều lượng (mg, ml,...)
- TRIỆU_CHỨNG: cảm giác bất thường của bệnh nhân (đau, khó thở, sốt, mệt...)
- CHẨN_ĐOÁN: tên bệnh, hội chứng (viêm, ung thư, tăng huyết áp...)
- TÊN_XÉT_NGHIỆM: xét nghiệm, thủ thuật, hình ảnh (chụp X-quang, siêu âm...)
- KẾT_QUẢ_XÉT_NGHIỆM: kết quả xét nghiệm (bình thường, 0.01, âm tính...)

Ví dụ:
Mention: "đau ngực" (classified as THUỐC) -> TRIỆU_CHỨNG (đau ngực là triệu chứng)
Mention: "điện tâm đồ" (classified as TRIỆU_CHỨNG) -> TÊN_XÉT_NGHIỆM (điện tâm đồ là xét nghiệm)
Mention: "troponin 0.01" (classified as TÊN_XÉT_NGHIỆM) -> KẾT_QUẢ_XÉT_NGHIỆM (đây là kết quả)
Mention: "tăng huyết áp" (classified as TRIỆU_CHỨNG) -> CHẨN_ĐOÁN (tăng huyết áp là bệnh)
Mention: "metoprolol 25mg" (classified as KẾT_QUẢ_XÉT_NGHIỆM) -> THUỐC (metoprolol là thuốc)

Return JSON: correct (true/false), corrected_type (string), reason (string)
"""

# Prompt for LLM reranking system
LINKER_RERANK_SYSTEM = """Bạn là chuyên gia tra cứu mã bệnh (ICD-10 / RxNorm).
Chọn mã phù hợp nhất cho mention từ danh sách candidates.

Ví dụ ICD-10:
Mention: "tăng huyết áp" -> chọn I10 (Tăng huyết áp nguyên phát)
Mention: "đái tháo đường type 2" -> chọn E11.9 (ĐTĐ type 2 không biến chứng)

Ví dụ RxNorm:
Mention: "metoprolol 25mg" -> chọn 6910 (Metoprolol)
Mention: "aspirin 325mg" -> chọn 721818 (Aspirin 325mg)

Now select for this {mtype} mention.
Return JSON with keys: selected (list of codes), reason (string)
"""

# Prompt for agentic linker (tool-calling loop)
LINKER_AGENT_SYSTEM = """Bạn là chuyên gia tra cứu mã bệnh (ICD-10 / RxNorm).
Bạn có các công cụ sau để tìm mã phù hợp:
{tools}

CURRENT TASK:
- Loại: {mtype}
- Mention: "{mention}"
- Context: "{context}"

Hướng dẫn:
1. Gọi công cụ search_icd10 hoặc search_rxnorm để tìm candidates
2. Nếu có nhiều candidates, gọi rerank để chọn
3. Khi đã chắc chắn, trả về kết quả cuối cùng

Ví dụ gọi công cụ: {{"thought": "Cần tìm mã ICD-10 cho tăng huyết áp", "tool": "search_icd10", "args": {{"query": "tăng huyết áp"}}}}
Ví dụ trả lời cuối: {{"thought": "I10 là mã phù hợp nhất", "final": true, "candidates": ["I10"], "reason": "I10 - Tăng huyết áp nguyên phát"}}

Trả về JSON với tool call hoặc final result.
"""

# Prompt for ICD-10 candidate reranking
ICD10_RERANK_PROMPT_TEMPLATE = """Select best ICD-10 code(s) from candidates.

Mention: "{mention}"
Context: "{context}"
Type: CHẨN_ĐOÁN

Ví dụ: "tăng huyết áp" -> I10 (Tăng huyết áp nguyên phát)
Ví dụ: "suy tim" -> I50.9 (Suy tim không đặc hiệu)

Candidates:
{candidates}

Return JSON with keys: selected (list), reason (string)
"""

# Prompt for RxNorm candidate reranking
RXNORM_RERANK_PROMPT_TEMPLATE = """Select best RxNorm code(s) from candidates.

Mention: "{mention}"
Context: "{context}"
Type: THUỐC

Ví dụ: "metoprolol 25mg" -> 6910 (Metoprolol)
Ví dụ: "aspirin 325mg" -> 721818 (Aspirin 325mg)

Candidates:
{candidates}

Return JSON with keys: selected (list), reason (string)
"""


# ─── Per-type extraction prompts ────────────────────────────────────

THUOC_EXTRACT_SYSTEM = "You are a medical entity extraction expert. Extract only medications. Return JSON array only."
THUOC_EXTRACT_USER = """Extract MEDICATION names from the following Vietnamese clinical text.

Rules:
- THUỐC: drug name + dose (if available). E.g., "metoprolol 25mg", "aspirin 325mg", "omeprazole 20mg"
- Extract ONLY drug names. DO NOT extract symptoms, diagnoses, or tests.
- Include dose (mg, ml, IU, mcg) when present.

Examples (Vietnamese):
Input: "Bệnh nhân dùng metoprolol 25mg bid, aspirin 325mg, và omeprazole."
Output: [{{"entity":"metoprolol 25mg","type":"THUỐC"}},{{"entity":"aspirin 325mg","type":"THUỐC"}},{{"entity":"omeprazole","type":"THUỐC"}}]

Input: "Tiêm insulin NPH 20 IU sáng và tối, kèm metformin 500mg."
Output: [{{"entity":"insulin NPH 20 IU","type":"THUỐC"}},{{"entity":"metformin 500mg","type":"THUỐC"}}]

Input: "Bệnh nhân được chỉ định amlodipine 5mg, enalapril 10mg."
Output: [{{"entity":"amlodipine 5mg","type":"THUỐC"}},{{"entity":"enalapril 10mg","type":"THUỐC"}}]

Text: "{text}"
Context: "{summary}"

Return JSON array: [{{"entity":"...", "type":"THUỐC"}}]"""

TRIEUCHUNG_EXTRACT_SYSTEM = "You are a medical entity extraction expert. Extract only symptoms. Return JSON array only."
TRIEUCHUNG_EXTRACT_USER = """Extract SYMPTOMS from the following Vietnamese clinical text.

Rules:
- TRIỆU_CHỨNG: patient complaints, abnormal sensations. E.g., pain, dyspnea, fever, nausea, fatigue, dizziness, cough
- Include negated symptoms (e.g., "không đau ngực", "không sốt")
- DO NOT extract medications, diagnoses, or test names
- Keep each symptom as a short phrase (2-6 words)

Examples (Vietnamese):
Input: "Bệnh nhân đau ngực trái, khó thở khi gắng sức, không sốt."
Output: [{{"entity":"đau ngực","type":"TRIỆU_CHỨNG"}},{{"entity":"khó thở khi gắng sức","type":"TRIỆU_CHỨNG"}},{{"entity":"không sốt","type":"TRIỆU_CHỨNG"}}]

Input: "Mệt mỏi nhiều, chán ăn, sút 5kg. Đau đầu vùng trán."
Output: [{{"entity":"mệt mỏi","type":"TRIỆU_CHỨNG"}},{{"entity":"chán ăn","type":"TRIỆU_CHỨNG"}},{{"entity":"đau đầu","type":"TRIỆU_CHỨNG"}}]

Input: "Buồn nôn và nôn 3 lần, đau bụng vùng thượng vị, không tiêu chảy."
Output: [{{"entity":"buồn nôn","type":"TRIỆU_CHỨNG"}},{{"entity":"nôn","type":"TRIỆU_CHỨNG"}},{{"entity":"đau bụng vùng thượng vị","type":"TRIỆU_CHỨNG"}},{{"entity":"không tiêu chảy","type":"TRIỆU_CHỨNG"}}]

Text: "{text}"
Context: "{summary}"

Return JSON array: [{{"entity":"...", "type":"TRIỆU_CHỨNG"}}]"""

CHANDOAN_EXTRACT_SYSTEM = "You are a medical entity extraction expert. Extract only diagnoses. Return JSON array only."
CHANDOAN_EXTRACT_USER = """Extract DIAGNOSES from the following Vietnamese clinical text.

Rules:
- CHẨN_ĐOÁN: disease names, syndromes, pathological conditions. E.g., hypertension, pneumonia, heart failure, diabetes type 2
- Include uncertain diagnoses (contains "có thể", "nghi ngờ")
- DO NOT extract symptoms, medications, or tests

Examples (Vietnamese):
Input: "Tiền sử tăng huyết áp, đái tháo đường type 2. Hiện tại viêm phổi cộng đồng."
Output: [{{"entity":"tăng huyết áp","type":"CHẨN_ĐOÁN"}},{{"entity":"đái tháo đường type 2","type":"CHẨN_ĐOÁN"}},{{"entity":"viêm phổi cộng đồng","type":"CHẨN_ĐOÁN"}}]

Input: "Chẩn đoán: suy tim sung huyết, rung nhĩ, tăng huyết áp nguyên phát."
Output: [{{"entity":"suy tim sung huyết","type":"CHẨN_ĐOÁN"}},{{"entity":"rung nhĩ","type":"CHẨN_ĐOÁN"}},{{"entity":"tăng huyết áp nguyên phát","type":"CHẨN_ĐOÁN"}}]

Input: "Nghi ngờ nhồi máu cơ tim cấp, có thể viêm phổi bệnh viện."
Output: [{{"entity":"nhồi máu cơ tim cấp","type":"CHẨN_ĐOÁN"}},{{"entity":"viêm phổi bệnh viện","type":"CHẨN_ĐOÁN"}}]

Text: "{text}"
Context: "{summary}"

Return JSON array: [{{"entity":"...", "type":"CHẨN_ĐOÁN"}}]"""

XETNGHIEM_EXTRACT_SYSTEM = "You are a medical entity extraction expert. Extract only tests and results. Return JSON array only."
XETNGHIEM_EXTRACT_USER = """Extract TESTS + RESULTS from the following Vietnamese clinical text.

Extract test names and their nearby results carefully.

Rules:
- TÊN_XÉT_NGHIỆM: test name or diagnostic imaging study. E.g., "điện tâm đồ", "chụp X-quang ngực", "CT scan", "công thức máu"
- KẾT_QUẢ_XÉT_NGHIỆM: test result, measured value, or finding. E.g., "bình thường", "Hb 13.5", "gan to", "0.01 ng/mL"
- If a test name and its result appear next to each other or immediately after one another, treat them as related but extract them as TWO separate entities in the same order as the original text.
- Do NOT merge a test name and its result into one span.
- Keep spans short, exact, and faithful to the source text.
- Verbs such as "định", "làm", "cho", "thực hiện" usually are not part of the entity span.
- Procedure/imaging verbs such as "chụp" may be kept only when they are naturally part of the test name in the source text.
- You may extract a test without an explicit result if the text only mentions the test name.
- DO NOT extract: surgeries, therapeutic procedures, medications, symptoms, or diagnoses

Examples (Vietnamese):
Input: "Điện tâm đồ: nhịp xoang. X-quang ngực: bình thường."
Output: [{{"entity":"điện tâm đồ","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"nhịp xoang","type":"KẾT_QUẢ_XÉT_NGHIỆM"}},{{"entity":"X-quang ngực","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"bình thường","type":"KẾT_QUẢ_XÉT_NGHIỆM"}}]

Input: "Công thức máu: Hb 13.5, WBC 7.2. Siêu âm tim: EF 55%."
Output: [{{"entity":"công thức máu","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"Hb 13.5","type":"KẾT_QUẢ_XÉT_NGHIỆM"}},{{"entity":"WBC 7.2","type":"KẾT_QUẢ_XÉT_NGHIỆM"}},{{"entity":"siêu âm tim","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"EF 55%","type":"KẾT_QUẢ_XÉT_NGHIỆM"}}]

Input: "Troponin: 0.02. X-quang ngực: không có viêm phổi."
Output: [{{"entity":"Troponin","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"0.02","type":"KẾT_QUẢ_XÉT_NGHIỆM"}},{{"entity":"X-quang ngực","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"không có viêm phổi","type":"KẾT_QUẢ_XÉT_NGHIỆM"}}]

Input: "Định siêu âm tim qua thành và chụp X-quang ngực. Phân tích nước tiểu không ghi bất thường."
Output: [{{"entity":"siêu âm tim qua thành","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"X-quang ngực","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"phân tích nước tiểu","type":"TÊN_XÉT_NGHIỆM"}},{{"entity":"không ghi bất thường","type":"KẾT_QUẢ_XÉT_NGHIỆM"}}]

Text: "{text}"
Context: "{summary}"

Return JSON array: [{{"entity":"...", "type":"TÊN_XÉT_NGHIỆM"}}, {{"entity":"...", "type":"KẾT_QUẢ_XÉT_NGHIỆM"}}]"""
