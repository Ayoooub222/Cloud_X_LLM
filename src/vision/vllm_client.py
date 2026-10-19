#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from openai import AsyncOpenAI
from pydantic import ValidationError

from src.schemas.extraction import ExtractionPage, fusionner_pages
from src.vision.image_prep import PageImage, charger_document
from src.vision.prompts import SYSTEM_PROMPT, construire_message_utilisateur

log = logging.getLogger(__name__)


class ExtractionEchouee(RuntimeError):
    pass


@dataclass(slots=True)
class ResultatExtraction:
    document: ExtractionPage
    pages: list[ExtractionPage]
    tokens_prompt: int
    tokens_sortie: int
    duree_s: float
    texte_natif: bool


class VllmApiClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str = "qwen3-vl-2b",
                 max_concurrence: int = 4, max_tokens: int = 2500, timeout_s: float = 45.0):
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=1)
        self._sem = asyncio.Semaphore(max_concurrence)   # aligné sur --max-num-seqs
        self._response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_doc_auto",
                "schema": ExtractionPage.model_json_schema(),
                "strict": True,
            },
        }

    async def warmup(self) -> None:
        """Compile la grammaire xgrammar au démarrage (sinon le 1er client paie ce coût)."""
        try:
            await self._client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": "Document vide."}],
                response_format=self._response_format, max_tokens=8, temperature=0.0)
        except Exception as e:  # le warmup ne doit jamais faire tomber l'app
            log.warning("warmup vLLM échoué: %s", e)

    async def _extraire_page(self, page: PageImage, nb_pages: int) -> tuple[ExtractionPage, int, int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": page.data_url}},
                {"type": "text", "text": construire_message_utilisateur(page.index, nb_pages, page.texte_natif)},
            ]},
        ]
        erreur = "inconnue"
        async with self._sem:
            for temperature in (0.0, 0.2):   # 2e essai légèrement perturbé pour sortir d'une boucle
                resp = await self._client.chat.completions.create(
                    model=self.model, messages=messages, response_format=self._response_format,
                    temperature=temperature, top_p=1.0, max_tokens=self.max_tokens, seed=1234)
                choix, usage = resp.choices[0], resp.usage
                if choix.finish_reason == "length":
                    erreur = f"JSON tronqué à {self.max_tokens} tokens"
                    continue
                try:
                    ext = ExtractionPage.model_validate_json(choix.message.content or "")
                    return ext, usage.prompt_tokens, usage.completion_tokens
                except ValidationError as e:   # filet de sécurité si la contrainte n'est pas appliquée
                    erreur = str(e)[:400]
        raise ExtractionEchouee(f"page {page.index + 1}: {erreur}")

    async def analyser(self, data: bytes) -> ResultatExtraction:
        t0 = time.perf_counter()
        pages_img = await asyncio.to_thread(charger_document, data)
        res = await asyncio.gather(*(self._extraire_page(p, len(pages_img)) for p in pages_img))
        extractions = [r[0] for r in res]
        return ResultatExtraction(
            document=fusionner_pages(extractions),
            pages=extractions,
            tokens_prompt=sum(r[1] for r in res),
            tokens_sortie=sum(r[2] for r in res),
            duree_s=round(time.perf_counter() - t0, 2),
            texte_natif=any(p.texte_natif for p in pages_img),
        )


async def _main() -> None:
    from src.utils.loader import Loader
    ap = argparse.ArgumentParser()
    ap.add_argument("fichier", type=Path)
    args = ap.parse_args()

    conf = Loader().load_conf()
    client = VllmApiClient(base_url=conf["url_api"], api_key=Loader().load_env(),
                           model=conf.get("model", "qwen3-vl-2b"))
    await client.warmup()
    res = await client.analyser(args.fichier.read_bytes())

    out = Path("results") / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"   # pas de ':' (NTFS)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "document": res.document.model_dump(mode="json"),
        "metriques": {"duree_s": res.duree_s, "tokens_prompt": res.tokens_prompt,
                      "tokens_sortie": res.tokens_sortie, "texte_natif": res.texte_natif},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{out}  |  {res.duree_s}s  |  in={res.tokens_prompt} out={res.tokens_sortie}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
