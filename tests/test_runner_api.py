"""Pure helpers of runner_api (no network)."""
from src.runner.runner_api import PROVIDERS, without_keys


def test_without_keys_every_depth():
    schema = {"type": "object", "properties": {"a": {"type": "array", "maxItems": 40, "items": {"maxItems": 2}}},
              "$defs": {"B": {"maxItems": 3, "type": "array"}}}
    out = without_keys(schema, {"maxItems"})
    assert "maxItems" not in str(out)
    assert out["properties"]["a"]["type"] == "array"
    assert "maxItems" in str(schema)  # the input is not modified


def test_providers_use_free_models_and_labels():
    assert PROVIDERS["groq-audio"].env_label == "api-groq"
    assert PROVIDERS["gemini-vision"].task == "vision"
