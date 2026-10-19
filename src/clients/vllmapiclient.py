"""Client vLLM vision (API OpenAI-compatible) avec structured output."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

from src.utils.loader import Loader


class VLLMAPICLIENT:
    SCHEMA_DIR = Path(__file__).resolve().parents[2]

    def __init__(self, api_key, key_env, pdf_path=None, schema_response: str = "test_response"):
        self.pdf_path = pdf_path
        self.schema_response = schema_response
        self.schema: dict = self._load_schema(schema_response)
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema.get("title", schema_response),
                "schema": self.schema,
                "strict": True,
            },
        }
        self._client = OpenAI(api_key=api_key, base_url=key_env)  # 1 seul client, connexions réutilisées
        self.model: str | None = None
        self._hyperparameters: dict | None = None
        self.last_usage = None
        self.last_page_sizes: list[tuple[int, int]] = []

    def _load_schema(self, name: str) -> dict:
        with open(self.SCHEMA_DIR / name, encoding="utf-8") as f:
            return json.load(f)

    @property
    def default_prompt(self) -> str:
        # Vrai JSON du schéma (pas le repr Python du response_format) : guide la sémantique,
        # la syntaxe est garantie par response_format (décodage contraint).
        return f"Follow exactly this JSON schema:\n{json.dumps(self.schema, ensure_ascii=False)}"

    def pdf_to_base64_images(self, pdf_path, dpi: int = 100) -> list[str]:
        images = []
        self.last_page_sizes = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
                self.last_page_sizes.append((pix.width, pix.height))
        return images

    def resolve_model(self) -> str:
        if self.model is None:
            self.model = self._client.models.list().data[0].id
        return self.model

    def hyperparameters(self) -> dict:
        if self._hyperparameters is None:  # lu 1 fois, hors de la zone chronométrée ensuite
            name = self.resolve_model().rsplit("/", 1)[-1]
            self._hyperparameters = Loader().load_conf()["hyperparameter"][name]
        return self._hyperparameters

    def complete_vision(self, image_b64: str, prompt: str | None = None):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt or self.default_prompt},
            ],
        }]
        response = self._client.chat.completions.create(
            model=self.resolve_model(),
            messages=messages,
            response_format=self.response_format,
            extra_body=self.hyperparameters(),
        )
        self.last_usage = response.usage
        return response
