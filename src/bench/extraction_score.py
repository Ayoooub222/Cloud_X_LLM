"""Field-level accuracy of a vision extraction against a known ground truth.

Both JSON objects are flattened into "path -> leaf value" pairs (list items keep their index:
pieces[0].designation). A field matches when the normalized values are equal. Fields present on
one side only count as mismatches, so missing and invented lines both lower the score.
"""
from __future__ import annotations

import re

from src.schemas.normalisation import parse_decimal_fr, parse_hours

AMOUNT_KEYS = frozenset({
    "quantite", "prix_unitaire_ht", "remise", "montant_ht", "taux_tva", "taux_horaire_ht",
    "total_ht", "total_tva", "total_ttc", "kilometrage",
})
HOURS_KEYS = frozenset({"temps"})
DIGITS_KEYS = frozenset({"siret", "code_postal"})
APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})
NULL_TOKENS = frozenset({"", "-", "–", "—", "n/a"})


def flatten(value, path: str = "") -> dict[str, object]:
    """Flatten nested dicts and lists into {"a.b[0].c": leaf}."""
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out |= flatten(child, f"{path}.{key}" if path else key)
        return out
    if isinstance(value, list):
        out = {}
        for i, child in enumerate(value):
            out |= flatten(child, f"{path}[{i}]")
        return out
    return {path: value}


def normalize(key: str, value):
    """Comparable form of a leaf. Empty strings and dashes count as null."""
    if isinstance(value, bool) or value is None:
        return value
    text = " ".join(str(value).translate(APOSTROPHES).split()).casefold()
    if text in NULL_TOKENS:
        return None
    if key in AMOUNT_KEYS:
        return parse_decimal_fr(text) if parse_decimal_fr(text) is not None else text
    if key in HOURS_KEYS:
        return parse_hours(text) if parse_hours(text) is not None else text
    if key in DIGITS_KEYS:
        return re.sub(r"\D", "", text) or None
    return text


def leaf_key(path: str) -> str:
    """Last field name of a path: 'pieces[2].montant_ht' -> 'montant_ht'."""
    return re.sub(r"\[\d+\]$", "", path.rsplit(".", 1)[-1])


def score(expected: dict, predicted: dict) -> dict[str, float]:
    """Accuracy over the union of fields of both sides."""
    exp, pred = flatten(expected), flatten(predicted)
    paths = exp.keys() | pred.keys()
    matched = sum(
        1 for p in paths
        if p in exp and p in pred and normalize(leaf_key(p), exp[p]) == normalize(leaf_key(p), pred[p])
    )
    return {
        "fields_total": len(paths),
        "fields_matched": matched,
        "field_accuracy": round(matched / len(paths), 4) if paths else 1.0,
    }
