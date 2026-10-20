"""Client for the vLLM vision server (OpenAI-compatible API) with structured JSON output."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pymupdf
from openai import OpenAI

from src.utils.loader import Loader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_schema(schema_file: str | Path) -> dict:
    """Load a JSON schema (path relative to the project root, or absolute)."""
    with open(PROJECT_ROOT / schema_file, encoding="utf-8") as f:
        return json.load(f)


def pdf_to_pages(pdf_path: str | Path, dpi: int = 100) -> list[dict]:
    """Render each PDF page as PNG. Return [{"b64": ..., "width": ..., "height": ...}]."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            pages.append({
                "b64": base64.b64encode(pix.tobytes("png")).decode("utf-8"),
                "width": pix.width,
                "height": pix.height,
            })
    return pages


class VisionClient:
    """Send one page image to vLLM and get a JSON answer that follows a schema."""

    def __init__(self, api_key: str, base_url: str, schema_file: str):
        self.schema = load_schema(schema_file)
        self.schema_name = self.schema.get("title", schema_file)
        # Constrained decoding: vLLM guarantees the JSON syntax
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": self.schema_name, "schema": self.schema, "strict": True},
        }
        # The schema is also put in the prompt: it guides the meaning of the fields
        self.default_prompt = f"Follow exactly this JSON schema:\n{json.dumps(self.schema, ensure_ascii=False)}"
        self.openai = OpenAI(api_key=api_key, base_url=base_url)  # one client, connections are reused
        self.model: str | None = None
        self.hyperparameters: dict = {}

    def connect(self) -> None:
        """Ask the server for its model name and load its hyperparameters.

        Call it once before timing anything.
        """
        self.model = self.openai.models.list().data[0].id
        short_name = self.model.rsplit("/", 1)[-1]
        self.hyperparameters = Loader().load_conf()["hyperparameter"][short_name]

    def complete(self, image_b64: str, prompt: str | None = None):
        """Send one image + prompt. Return the raw OpenAI response (usage included)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt or self.default_prompt},
            ],
        }]
        return self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=self.response_format,
            extra_body=self.hyperparameters,
        )
