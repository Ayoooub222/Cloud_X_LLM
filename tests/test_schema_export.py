"""The versioned schema file must stay in sync with the Pydantic model."""
from src.schemas.export_schema import DEFAULT_OUT, schema_text


def test_versioned_schema_matches_model():
    assert DEFAULT_OUT.read_text(encoding="utf-8") == schema_text(), "run: python -m src.schemas.export_schema"
