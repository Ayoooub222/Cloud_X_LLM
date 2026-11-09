"""The fictitious quote: ground truth consistent and file output reproducible."""
import json
from decimal import Decimal

from src.bench.synthetic_quote import DEFAULT_PDF, expected_page, parse_fr, quote_html, render_pdf


def test_totals_add_up():
    page = expected_page()
    lines = page["pieces"] + page["main_oeuvre"] + page["frais_divers"]
    total_ht = sum((parse_fr(line["montant_ht"]) for line in lines), Decimal(0))
    totals = page["totaux"]
    assert parse_fr(totals["total_ht"]) == total_ht
    assert parse_fr(totals["total_ht"]) + parse_fr(totals["total_tva"]) == parse_fr(totals["total_ttc"])


def test_versioned_ground_truth_matches_generator():
    versioned = json.loads(DEFAULT_PDF.with_suffix(".expected.json").read_text(encoding="utf-8"))
    assert versioned == [expected_page()], "run: python -m src.bench.synthetic_quote"


def test_pdf_is_deterministic_and_versioned():
    pdf = render_pdf(quote_html(expected_page()))
    assert pdf == render_pdf(quote_html(expected_page()))
    assert pdf == DEFAULT_PDF.read_bytes(), "run: python -m src.bench.synthetic_quote"
