import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class FakeExtractionLLM:
    def is_available(self):
        return True

    def generate(self, system_prompt, user_prompt, max_tokens=512):
        return '.JSON\n[{"entity":"đau ngực","type":"TRIỆU_CHỨNG"}]'

    def parse_json(self, response):
        from vllm_llm import VLLMLLM
        return VLLMLLM.parse_json(response)


class FakeDrugLLM(FakeExtractionLLM):
    def generate(self, system_prompt, user_prompt, max_tokens=512):
        return '[{"entity":"aspirin 81 mg","type":"THUỐC"}]'


def test_vllm_parser_accepts_qwen_json_prefix():
    from vllm_llm import VLLMLLM

    value = VLLMLLM.parse_json('.JSON\n[{"entity":"ho","type":"TRIỆU_CHỨNG"}]')
    assert value == [{"entity": "ho", "type": "TRIỆU_CHỨNG"}]


def test_qwen_span_is_expanded_to_distinct_occurrences():
    from ner_tool import ner_and_type

    text = "Bệnh nhân đau ngực, sau đó lại đau ngực."
    mentions = ner_and_type(text, llm=FakeExtractionLLM())["mentions"]

    assert [mention["text"] for mention in mentions] == ["đau ngực", "đau ngực"]
    assert [mention["position"] for mention in mentions] == [[10, 18], [31, 39]]


def test_short_mentions_are_not_expanded_inside_other_words():
    from ner_tool import _rule_find_positions

    text = "bactrim cho thấy lactate tăng; chụp ct ngực; bệnh nhân ho"
    ct_start = text.index("ct ngực")
    ho_start = text.rindex("ho")
    assert _rule_find_positions("ct", text) == [[ct_start, ct_start + 2]]
    assert _rule_find_positions("ho", text) == [[ho_start, ho_start + 2]]


def test_long_glued_entity_is_allowed_as_last_resort():
    from ner_tool import _rule_find_positions

    text = "đã sử dụng atenololtrong ngày"
    start = text.index("atenolol")
    assert _rule_find_positions("atenolol", text) == [[start, start + len("atenolol")]]


def test_type_rules_fix_vitals_labs_and_obvious_fragments():
    from ner_tool import _rule_verify_type

    assert _rule_verify_type("HR 140", "TRIỆU_CHỨNG") == "KẾT_QUẢ_XÉT_NGHIỆM"
    assert _rule_verify_type("hạ huyết áp", "TRIỆU_CHỨNG") == "KẾT_QUẢ_XÉT_NGHIỆM"
    assert _rule_verify_type("kali", "TRIỆU_CHỨNG") == "TÊN_XÉT_NGHIỆM"
    assert _rule_verify_type("viêm túi mật", "TRIỆU_CHỨNG") == "CHẨN_ĐOÁN"
    assert _rule_verify_type("tiếp", "TRIỆU_CHỨNG") is None


def test_drug_span_extends_through_route_frequency_and_prn():
    from ner_tool import ner_and_type

    text = "aspirin 81 mg po q6h:prn điều trị đau"
    mentions = ner_and_type(text, llm=FakeDrugLLM())["mentions"]

    assert mentions[0]["text"] == "aspirin 81 mg po q6h:prn"
    assert text[slice(*mentions[0]["position"])] == mentions[0]["text"]


def test_rule_extracts_medication_indication_symptom():
    from tools import rule_extract

    text = "aspirin 81 mg po daily điều trị đau ngực 2. thuốc khác"
    mentions = rule_extract(text)["mentions"]

    assert any(m["text"] == "đau ngực" and m["type"] == "TRIỆU_CHỨNG" for m in mentions)


def test_rule_splits_organizer_adjacent_anxiety_insomnia():
    from tools import rule_extract

    text = "clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ"
    symptoms = [m["text"] for m in rule_extract(text)["mentions"] if m["type"] == "TRIỆU_CHỨNG"]

    assert symptoms == ["lo âu", "mất ngủ"]


def test_voting_keeps_two_atomic_rule_spans_over_one_broad_qwen_span():
    from voting import merge_with_voting

    mentions = [
        {"text": "lo âu mất ngủ", "position": [0, 13], "type": "TRIỆU_CHỨNG", "source": "qwen"},
        {"text": "lo âu", "position": [0, 5], "type": "TRIỆU_CHỨNG", "source": "rule"},
        {"text": "mất ngủ", "position": [6, 13], "type": "TRIỆU_CHỨNG", "source": "rule"},
    ]

    merged = merge_with_voting(mentions)
    assert [(m["text"], m["position"]) for m in merged] == [
        ("lo âu", [0, 5]), ("mất ngủ", [6, 13])
    ]


def test_position_fix_preserves_second_repeated_occurrence():
    from position_fix import fix_positions

    text = "ho rồi hết, sau đó ho lại"
    second_start = text.rfind("ho")
    mentions = [{"text": "ho", "position": [second_start, second_start + 2], "type": "TRIỆU_CHỨNG"}]
    fixed = fix_positions(mentions, text)

    assert fixed[0]["position"] == [second_start, second_start + 2]
    assert text[second_start:second_start + 2] == fixed[0]["text"]


def test_rule_drug_text_matches_its_position():
    from tools import rule_extract

    text = "Dùng metoprolol 25mg po bid và theo dõi."
    drugs = [m for m in rule_extract(text)["mentions"] if m["type"] == "THUỐC"]

    assert drugs
    for drug in drugs:
        assert text[slice(*drug["position"])] == drug["text"]


def test_output_contract_uses_assertion_list_and_one_candidate():
    from agents import AgentContext

    ctx = AgentContext("tiền sử tăng huyết áp")
    ctx.mentions = [{
        "text": "tăng huyết áp",
        "position": [9, 23],
        "type": "CHẨN_ĐOÁN",
        "assertions": {"isNegated": False, "isFamily": False, "isHistorical": True},
        "candidates": ["I10", "I11"],
        "confidence": 0.9,
        "voters": {"qwen": 1},
    }]

    output = ctx.to_output()
    assert output == [{
        "text": "tăng huyết áp",
        "type": "CHẨN_ĐOÁN",
        "assertions": ["isHistorical"],
        "position": [9, 23],
        "candidates": ["I10"],
    }]


def test_assertion_rules_respect_section_and_clause_scope():
    from agents.worker_agents import AssertionAgent

    agent = AssertionAgent()
    text = (
        "Tiền sử bệnh\nThuốc trước khi nhập viện\n- aspirin 81mg\n"
        "Các yếu tố nguy cơ liên quan\n- Theo lời bệnh nhân: căng thẳng\n"
        "Hiện tại không đau ngực nhưng khó thở"
    )
    aspirin = text.index("aspirin")
    stress = text.index("căng thẳng")
    chest_pain = text.index("đau ngực")
    dyspnea = text.index("khó thở")

    assert agent._rule_assert(text, aspirin, aspirin + len("aspirin"))["isHistorical"]
    stress_assertions = agent._rule_assert(text, stress, stress + len("căng thẳng"))
    assert not stress_assertions["isFamily"]
    assert not stress_assertions["isHistorical"]
    assert agent._rule_assert(text, chest_pain, chest_pain + len("đau ngực"))["isNegated"]
    assert not agent._rule_assert(text, dyspnea, dyspnea + len("khó thở"))["isNegated"]

    medication_list = "Danh sách thuốc trước nhập viện: thuốc điều trị lo âu mất ngủ"
    pain = medication_list.index("mất ngủ")
    assert not agent._rule_assert(
        medication_list, pain, pain + len("mất ngủ"), "TRIỆU_CHỨNG"
    )["isHistorical"]

    current_note = "Lịch sử bệnh hiện tại\nLý do nhập viện: đau bụng"
    abdominal_pain = current_note.index("đau bụng")
    assert not agent._rule_assert(
        current_note, abdominal_pain, abdominal_pain + len("đau bụng"), "TRIỆU_CHỨNG"
    )["isHistorical"]


def test_assertion_rules_cover_denial_and_negative_results():
    from agents.worker_agents import AssertionAgent

    agent = AssertionAgent()
    denial = "Bệnh nhân phủ nhận khó thở, đau ngực"
    dyspnea = denial.index("khó thở")
    chest_pain = denial.index("đau ngực")
    assert agent._rule_assert(denial, dyspnea, dyspnea + 7)["isNegated"]
    assert agent._rule_assert(denial, chest_pain, chest_pain + 8)["isNegated"]

    negative_scan = "Siêu âm âm tính với huyết khối tĩnh mạch sâu"
    thrombosis = negative_scan.index("huyết khối")
    assert agent._rule_assert(
        negative_scan, thrombosis, len(negative_scan), "CHẨN_ĐOÁN"
    )["isNegated"]


def test_verifier_keeps_distinct_and_nested_spans():
    from agents import AgentContext
    from agents.worker_agents import VerifierAgent

    text = "đau ngực rồi đau ngực"
    second_start = text.rfind("đau ngực")
    base = {"type": "TRIỆU_CHỨNG", "assertions": {}, "confidence": 0.8}
    ctx = AgentContext(text)
    ctx.normalized = text
    ctx.mentions = [
        {**base, "text": "đau ngực", "position": [0, 8]},
        {**base, "text": "đau", "position": [0, 3]},
        {**base, "text": "đau ngực", "position": [second_start, second_start + 8]},
    ]

    result = VerifierAgent().run(ctx)
    assert [(m["text"], m["position"]) for m in result.mentions] == [
        ("đau", [0, 3]),
        ("đau ngực", [0, 8]),
        ("đau ngực", [second_start, second_start + 8]),
    ]


def test_linker_fallback_emits_only_top_one(monkeypatch):
    import linker_agent

    class KB:
        def search_icd10(self, query, top_k=15):
            return [
                {"code": "I10", "name": "Tăng huyết áp", "score": 1.0},
                {"code": "I11", "name": "Bệnh tim do tăng huyết áp", "score": 0.8},
            ]

    monkeypatch.setattr(linker_agent, "_embedding_rerank", lambda text, mtype, candidates, kb: candidates)
    result = linker_agent.link_candidates(
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"},
        "Chẩn đoán tăng huyết áp",
        KB(),
        llm=None,
    )
    assert result["candidates"] == ["I10"]


def test_reviewed_icd10_aliases_override_fuzzy_linking():
    from linker_agent import link_candidates

    result = link_candidates(
        {"text": "xơ gan do rượu", "type": "CHẨN_ĐOÁN"},
        "Tiền sử xơ gan do rượu",
        object(),
        llm=None,
    )
    assert result == {"candidates": ["K70.3"], "confidence": 0.99}


def test_round2_reviewed_icd10_aliases_override_bad_fuzzy_codes():
    from linker_agent import link_candidates

    result = link_candidates(
        {"text": "nhiễm khuẩn đường tiết niệu", "type": "CHẨN_ĐOÁN"},
        "Chẩn đoán nhiễm khuẩn đường tiết niệu",
        object(),
        llm=None,
    )
    assert result == {"candidates": ["N39.0"], "confidence": 0.99}


def test_round2_reviewed_rxnorm_aliases_link_missing_brand():
    from linker_agent import link_candidates

    result = link_candidates(
        {"text": "bactrim", "type": "THUỐC"},
        "Thuốc trước nhập viện: bactrim",
        object(),
        llm=None,
    )
    assert result == {"candidates": ["151399"], "confidence": 0.99}


def test_rxnorm_structured_retrieval_matches_organizer_examples():
    from kb import KnowledgeBase
    from linker_agent import _search_rxnorm

    kb = KnowledgeBase()
    kb.load()
    examples = {
        "amlodipine 10 mg po daily": "308135",
        "aspirin 81 mg po daily": "243670",
        "metoprolol succinate xl 50 mg po daily": "866436",
        "guaifenesin ml po q6h:prn": "392085",
        "nystatin oral suspension 5 ml po qid:prn": "7597",
        "acetaminophen 325-650 mg po q6h:prn": "313782",
        "pravastatin 40 mg po daily": "904475",
        "docusate sodium 100 mg po bid": "1099279",
        "senna 8.6 mg po bid:prn": "312935",
        "clonazepam 0.5 mg po qam:prn": "197527",
        "clonazepam 1.5 mg po qhs": "197528",
    }

    for mention, expected in examples.items():
        candidates = _search_rxnorm(mention, kb, top_k=5)
        assert candidates[0]["code"] == expected, (mention, candidates[:3])


def test_rxnorm_retrieval_respects_intravenous_route_and_total_dose():
    from kb import KnowledgeBase
    from linker_agent import _search_rxnorm

    kb = KnowledgeBase()
    kb.load()
    examples = {
        "levofloxacin 750mg iv": "1665519",
        "metoprolol 5mg iv x2": "866508",
        "Laxis 20mg tiêm tĩnh mạch": "1719290",
        "ceftriaxone 1 gram": "1665021",
    }
    for mention, expected in examples.items():
        candidates = _search_rxnorm(mention, kb, top_k=5)
        assert candidates[0]["code"] == expected, (mention, candidates[:3])

    bumetanide = _search_rxnorm("bumetanide 2mg iv", kb, top_k=5)[0]
    assert "injection" in bumetanide["name"].casefold()
