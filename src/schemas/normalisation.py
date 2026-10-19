"""Parsing déterministe des chaînes brutes extraites par le VLM. Aucune heuristique 'intelligente'."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_RE_NUM = re.compile(r"-?\d[\d\s\u00a0\u202f.,']*")
_RE_CT = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{1,2}(\.[a-z](\.\d{1,2})?)?$")
_RE_CP = re.compile(r"\b(\d{5})\b")
_RE_SIV = re.compile(r"^([A-Z]{2})(\d{3})([A-Z]{2})$")


def parse_decimal_fr(brut: str | None) -> Decimal | None:
    if not brut:
        return None
    m = _RE_NUM.search(brut.replace("€", ""))
    if not m:
        return None
    t = re.sub(r"[\s\u00a0\u202f']", "", m.group()).rstrip(".,")
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif t.count(",") == 1:
        t = t.replace(",", ".")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def parse_heures(brut: str | None) -> Decimal | None:
    if not brut:
        return None
    s = brut.lower().replace(" ", "")
    if m := re.fullmatch(r"(\d+)h(\d{1,2})?(min)?", s):
        return (Decimal(m[1]) + Decimal(m[2] or 0) / 60).quantize(Decimal("0.01"))
    if m := re.fullmatch(r"(\d+)(min|mn)", s):
        return (Decimal(m[1]) / 60).quantize(Decimal("0.01"))
    return parse_decimal_fr(s)


def valider_siret(brut: str | None) -> str | None:
    """SIREN (9) ou SIRET (14), contrôle de Luhn : rejette les SIRET hallucinés."""
    chiffres = re.sub(r"\D", "", brut or "")
    if len(chiffres) not in (9, 14):
        return None
    total = 0
    for i, c in enumerate(reversed(chiffres)):
        d = int(c) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return chiffres if total % 10 == 0 or chiffres.startswith("356000000") else None


def normaliser_code_ct(brut: str | None) -> str | None:
    if not brut:
        return None
    s = re.sub(r"[\s,;]+", ".", brut.strip().lower()).strip(".")
    s = re.sub(r"\.{2,}", ".", s)
    return s if _RE_CT.match(s) else None


def normaliser_code_postal(cp: str | None, adresse: str | None) -> str | None:
    for src in (cp, adresse):
        if src and (m := _RE_CP.search(src)):
            return m[1]
    return None


def normaliser_immat(brut: str | None) -> str | None:
    s = re.sub(r"[^A-Z0-9]", "", (brut or "").upper())
    m = _RE_SIV.match(s)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else (s or None)
