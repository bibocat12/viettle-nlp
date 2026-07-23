import json

from scripts.reward_v2 import parse_completion, score_entities


def entity(text="đau ngực", entity_type="TRIỆU_CHỨNG", position=None, **extra):
    value = {
        "entity": text,
        "type": entity_type,
        "position": position or [0, len(text)],
        "assertions": [],
        "candidates": [],
    }
    value.update(extra)
    return value


def test_exact_prediction_is_perfect():
    truth = [entity()]
    assert score_entities(truth, truth)["score"] == 1.0


def test_wrong_type_is_penalized_as_unmatched():
    truth = [entity()]
    prediction = [entity(entity_type="CHẨN_ĐOÁN")]
    assert score_entities(prediction, truth)["text"] == 0.0


def test_boundary_error_gets_partial_shaping_reward():
    truth = [entity("đau ngực dữ dội", position=[10, 26])]
    prediction = [entity("đau ngực", position=[10, 18])]
    result = score_entities(prediction, truth)
    assert 0.0 < result["text"] < 1.0


def test_unsupervised_candidate_does_not_force_empty_code():
    truth = [entity(
        "bệnh hiếm", "CHẨN_ĐOÁN", candidates=[], _candidate_supervised=False
    )]
    prediction = [entity("bệnh hiếm", "CHẨN_ĐOÁN", candidates=["A00"])]
    assert score_entities(prediction, truth)["candidates"] == 1.0


def test_extra_entity_reduces_score():
    truth = [entity()]
    prediction = [entity(), entity("sốt", position=[20, 23])]
    assert score_entities(prediction, truth)["score"] < 1.0


def test_parser_tolerates_prefix():
    content = "JSON:\n" + json.dumps([entity()], ensure_ascii=False)
    assert parse_completion(content)[0]["entity"] == "đau ngực"


def test_object_assertions_are_scored_semantically_but_schema_penalized():
    truth = [entity(assertions=["isNegated"])]
    prediction = [entity(assertions={"isNegated": True})]

    result = score_entities(prediction, truth)

    assert result["assertions"] == 1.0
    assert result["schema"] == 0.0


def test_dict_inside_assertion_list_is_invalid_without_crashing_reward():
    truth = [entity(assertions=[])]
    prediction = [entity(assertions=[{"isHistorical": False}])]

    result = score_entities(prediction, truth)

    assert result["assertions"] == 0.0
    assert result["schema"] == 0.0
