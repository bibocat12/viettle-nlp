import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_schema_client_sends_only_entity_and_type(monkeypatch):
    import requests
    from grpo_v2_fusion import PROPOSAL_SCHEMA
    from vllm_llm import VLLMLLM

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    client = VLLMLLM(max_retries=1)
    assert client.generate_json("system", "user", schema=PROPOSAL_SCHEMA) == {"entities": []}
    schema = captured["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]["entities"]["items"]["properties"]) == {"entity", "type"}
    assert captured["temperature"] == 0.0
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_alignment_ignores_model_offsets_and_keeps_repeated_occurrences():
    from grpo_v2_fusion import _align_proposals

    text = "đau ngực, sau đó lại đau ngực"
    proposals = _align_proposals(
        [{"entity": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [999, 1000]}],
        text,
        0,
        "general",
    )
    assert [item["position"] for item in proposals] == [[0, 8], [21, 29]]
    assert all(text[slice(*item["position"])] == item["text"] for item in proposals)


def test_chunk_cache_resumes_without_calling_model(tmp_path):
    from grpo_v2_fusion import extract_pass

    class Client:
        def __init__(self):
            self.calls = 0

        def generate_json(self, *args, **kwargs):
            self.calls += 1
            return {"entities": [{"entity": "đau ngực", "type": "TRIỆU_CHỨNG"}]}

    client = Client()
    first, first_events = extract_pass("record", "đau ngực", "general", client, tmp_path, resume=False)
    second, second_events = extract_pass("record", "đau ngực", "general", client, tmp_path, resume=True)
    assert client.calls == 1
    assert first == second
    assert first_events[0]["status"] == "ok"
    assert second_events[0]["status"] == "ok"
    assert (tmp_path / "record" / "general-0.jsonl").exists()


def test_fusion_keeps_anchor_and_rejects_unlinked_new_diagnosis(monkeypatch):
    from grpo_v2_fusion import FusionSettings, _fuse

    class KB:
        icd10 = {"I10": {"code": "I10"}}
        rxnorm_concepts = {}

    monkeypatch.setattr("grpo_v2_fusion.rule_extract", lambda _: {"mentions": []})
    monkeypatch.setattr("grpo_v2_fusion._candidate_for", lambda entity, kb, settings: ([], "no_kb_match"))
    text = "đau ngực và bệnh lạ"
    baseline = [{"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [0, 8]}]
    proposals = [
        {"text": "bệnh lạ", "type": "CHẨN_ĐOÁN", "position": [13, 20], "pass_id": pass_id}
        for pass_id in ("general", "clinical")
    ]
    result, audit = _fuse(text, baseline, proposals, KB(), FusionSettings())
    assert result == baseline
    assert any(item["reason"] == "no_valid_kb_code" for item in audit if item["action"] == "rejected")


def test_fusion_adds_consensus_with_valid_code_and_strips_audit_fields(monkeypatch):
    from grpo_v2_fusion import FusionSettings, _fuse

    class KB:
        icd10 = {"I10": {"code": "I10"}}
        rxnorm_concepts = {}

    monkeypatch.setattr("grpo_v2_fusion.rule_extract", lambda _: {"mentions": []})
    monkeypatch.setattr("grpo_v2_fusion._candidate_for", lambda entity, kb, settings: (["I10"], "reviewed_alias"))
    text = "tăng huyết áp"
    proposals = [
        {"text": text, "type": "CHẨN_ĐOÁN", "position": [0, len(text)], "pass_id": pass_id}
        for pass_id in ("general", "clinical")
    ]
    result, _ = _fuse(text, [], proposals, KB(), FusionSettings())
    assert result == [{"text": text, "type": "CHẨN_ĐOÁN", "position": [0, len(text)], "assertions": [], "candidates": ["I10"]}]
    assert set(result[0]) == {"text", "type", "position", "assertions", "candidates"}
