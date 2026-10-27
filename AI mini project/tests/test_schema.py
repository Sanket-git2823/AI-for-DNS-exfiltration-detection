"""
Unit tests for schema mapping module.
"""
import pytest
import pandas as pd
from src.schema_mapping import SchemaMapper, REQUIRED_CANONICAL_FIELDS


def test_auto_detect_standard_columns():
    columns = ["query_name", "timestamp", "client_ip", "record_type", "response_code", "label"]
    mapper = SchemaMapper.auto_detect(columns)
    assert mapper.mapping["query_name"] == "query_name"
    assert mapper.mapping["timestamp"] == "timestamp"
    assert mapper.mapping["label"] == "label"


def test_auto_detect_nonstandard_columns():
    columns = ["domain", "src_ip", "qtype", "ts", "is_attack"]
    mapper = SchemaMapper.auto_detect(columns)
    assert mapper.mapping["query_name"] == "domain"
    assert mapper.mapping["client_ip"] == "src_ip"
    assert mapper.mapping["record_type"] == "qtype"
    assert mapper.mapping["timestamp"] == "ts"
    assert mapper.mapping["label"] == "is_attack"


def test_validate_mapping_success():
    df = pd.DataFrame({"query_name": ["google.com"], "label": [0]})
    mapper = SchemaMapper(mapping={"query_name": "query_name", "label": "label"})
    valid, errors = mapper.validate_mapping(df)
    assert valid is True
    assert len(errors) == 0


def test_validate_mapping_missing_required():
    df = pd.DataFrame({"something": ["a"]})
    mapper = SchemaMapper(mapping={"query_name": None})
    valid, errors = mapper.validate_mapping(df)
    assert valid is False
    assert any("query_name" in e for e in errors)


def test_apply_mapping():
    df = pd.DataFrame({
        "domain": ["WWW.Google.Com.", "api.GITHUB.com"],
        "is_attack": ["benign", "malicious"]
    })
    mapper = SchemaMapper(mapping={"query_name": "domain", "label": "is_attack"})
    norm = mapper.apply_mapping(df)
    assert norm["query_name"].iloc[0] == "www.google.com"
    assert norm["query_name"].iloc[1] == "api.github.com"
    assert norm["label"].iloc[0] == 0
    assert norm["label"].iloc[1] == 1


def test_normalize_labels():
    series = pd.Series(["benign", "malicious", "1", "0", "attack", "normal", "True", "False"])
    result = SchemaMapper._normalize_labels(series)
    assert list(result) == [0, 1, 1, 0, 1, 0, 1, 0]
