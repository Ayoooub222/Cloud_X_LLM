"""Deterministic parsing of the raw strings extracted by the VLM. No 'smart' heuristics."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_RE = re.compile(r"-?\d[\d\s\u00a0\u202f.,']*")
CT_CODE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{1,2}(\.[a-z](\.\d{1,2})?)?$")  # e.g. 5.2.3.a.1
POSTCODE_RE = re.compile(r"\b(\d{5})\b")
PLATE_RE = re.compile(r"^([A-Z]{2})(\d{3})([A-Z]{2})$")  # French SIV plate: AB123CD


def parse_decimal_fr(raw: str | None) -> Decimal | None:
    """Parse an amount written the French way ('1 234,56 €', '1.234,56', '12,5')."""
    if not raw:
        return None
    match = NUMBER_RE.search(raw.replace("€", ""))
    if not match:
        return None
    text = re.sub(r"[\s\u00a0\u202f']", "", match.group()).rstrip(".,")

    if "," in text and "." in text:
        # Both separators: the last one is the decimal separator
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") == 1:
        text = text.replace(",", ".")
    elif text.count(".") > 1:  # dots used as thousands separators
        text = text.replace(".", "")

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_hours(raw: str | None) -> Decimal | None:
    """Parse a labour time ('1h30', '45min', '1,5') into decimal hours."""
    if not raw:
        return None
    text = raw.lower().replace(" ", "")
    if m := re.fullmatch(r"(\d+)h(\d{1,2})?(min)?", text):
        return (Decimal(m[1]) + Decimal(m[2] or 0) / 60).quantize(Decimal("0.01"))
    if m := re.fullmatch(r"(\d+)(min|mn)", text):
        return (Decimal(m[1]) / 60).quantize(Decimal("0.01"))
    return parse_decimal_fr(text)


def validate_siret(raw: str | None) -> str | None:
    """SIREN (9 digits) or SIRET (14 digits) with a Luhn check: rejects hallucinated numbers.

    La Poste SIRETs (starting with 356000000) do not follow Luhn and are accepted as is.
    """
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) not in (9, 14):
        return None
    total = 0
    for i, char in enumerate(reversed(digits)):
        value = int(char) * (2 if i % 2 else 1)
        total += value - 9 if value > 9 else value
    return digits if total % 10 == 0 or digits.startswith("356000000") else None


def normalize_ct_code(raw: str | None) -> str | None:
    """Normalize a vehicle inspection defect code ('5 2 3 a' -> '5.2.3.a'). None if invalid."""
    if not raw:
        return None
    text = re.sub(r"[\s,;]+", ".", raw.strip().lower()).strip(".")
    text = re.sub(r"\.{2,}", ".", text)
    return text if CT_CODE_RE.match(text) else None


def normalize_postcode(postcode: str | None, address: str | None) -> str | None:
    """5-digit postcode, taken from the postcode field or else from the address."""
    for source in (postcode, address):
        if source and (m := POSTCODE_RE.search(source)):
            return m[1]
    return None


def normalize_plate(raw: str | None) -> str | None:
    """'ab 123 cd' -> 'AB-123-CD'. Other formats are returned cleaned (uppercase, alphanumeric)."""
    text = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    m = PLATE_RE.match(text)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else (text or None)
