"""Build a fictitious one-page French garage quote (PDF) and its ground truth (JSON).

The PDF and the expected extraction come from the same data, so they cannot drift apart.
Every identifier is fake on purpose (SIRET 000..., specimen plate AA-123-AA, TEST VIN,
01 99 00 phone range reserved for fiction). The output is deterministic: same bytes on every run.

Example:
    python -m src.bench.synthetic_quote
"""
from __future__ import annotations

import argparse
import html
import io
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pymupdf

from src.schemas.extraction import ExtractionPage

DEFAULT_PDF = Path("inputs/vision/devis_fictif.pdf")
PAGE_MARGIN_PT = 36
VAT_RATE = Decimal("20")
CENT = Decimal("0.01")
EMPTY_METADATA = dict.fromkeys(("creationDate", "modDate", "producer", "creator", "title", "author"), "")

GARAGE = {
    "nom": "Garage des Essais (fictif)",
    "adresse": "12 rue des Essais",
    "code_postal": "99000",
    "ville": "Villetest",
    "siret": "000 000 000 00000",
}
VEHICLE = {
    "immatriculation": "AA-123-AA",
    "marque_modele": "Peugeot 208 1.2 PureTech",
    "vin": "VF3TEST0000000001",
    "kilometrage": "84 512 km",
    "date_mise_circulation": "14/06/2019",
}
HEADER = {"numero_document": "D-2026-0042", "date_document": "03/11/2026", "date_fin_validite": "03/12/2026"}
HOURLY_RATES = {"T1": Decimal("58.00"), "T2": Decimal("68.00")}

# reference, designation, quantity, unit price excl. VAT, discount %
PARTS = [
    ("PLQ-208-AV", "Jeu de plaquettes de frein avant", 1, "48.90", None),
    ("DSQ-208-AV", "Disques de frein avant (paire)", 1, "96.50", "10"),
    ("FH-1103", "Filtre à huile", 1, "12.40", None),
    ("HM-5W30-5L", "Huile moteur 5W30 (bidon 5 L)", 1, "42.00", None),
    ("FA-208", "Filtre à air", 1, "18.75", None),
    ("BEG-650", "Balais d'essuie-glace (paire)", 1, "24.90", "5"),
    ("AMP-H7", "Ampoule H7 12 V", 2, "6.90", None),
]
# designation, category, hours
LABOUR = [
    ("Remplacement disques et plaquettes avant", "T2", "1.5"),
    ("Vidange et remplacement des filtres", "T1", "0.8"),
    ("Remplacement balais et ampoules", "T1", "0.3"),
]
FEES = [("Forfait recyclage des déchets", "8.50"), ("Produits d'atelier", "5.90")]


def fr(amount: Decimal) -> str:
    """French amount: 1234.5 -> '1 234,50'."""
    return f"{amount.quantize(CENT, ROUND_HALF_UP):,.2f}".replace(",", " ").replace(".", ",")


def part_lines() -> list[dict]:
    lines = []
    for ref, label, qty, unit, discount in PARTS:
        gross = Decimal(unit) * qty
        net = gross * (1 - Decimal(discount or 0) / 100)
        lines.append({
            "reference": ref, "designation": label, "quantite": str(qty), "prix_unitaire_ht": fr(Decimal(unit)),
            "remise": f"{discount} %" if discount else None, "montant_ht": fr(net), "taux_tva": f"{VAT_RATE} %",
        })
    return lines


def labour_lines() -> list[dict]:
    return [{
        "designation": label, "categorie": cat, "temps": f"{hours.replace('.', ',')} h",
        "taux_horaire_ht": fr(HOURLY_RATES[cat]), "montant_ht": fr(Decimal(hours) * HOURLY_RATES[cat]),
    } for label, cat, hours in LABOUR]


def fee_lines() -> list[dict]:
    return [{"designation": label, "montant_ht": fr(Decimal(amount))} for label, amount in FEES]


def parse_fr(text: str) -> Decimal:
    return Decimal(text.replace(" ", "").replace(",", "."))


def totals(lines: list[dict]) -> dict:
    total_ht = sum((parse_fr(line["montant_ht"]) for line in lines), Decimal(0))
    vat = (total_ht * VAT_RATE / 100).quantize(CENT, ROUND_HALF_UP)
    return {"total_ht": fr(total_ht), "taux_tva": f"{VAT_RATE} %", "total_tva": fr(vat),
            "total_ttc": fr(total_ht + vat)}


def expected_page() -> dict:
    """Ground truth, validated against the extraction schema."""
    parts, labour, fees = part_lines(), labour_lines(), fee_lines()
    data = {
        "type_document": "DEVIS_GARAGE", "lisibilite": "BONNE",
        "emetteur": GARAGE, "vehicule": VEHICLE, "en_tete": HEADER,
        "pieces": parts, "main_oeuvre": labour, "frais_divers": fees,
        "resultat_ct": "NON_APPLICABLE", "defauts_ct": [],
        "totaux": totals(parts + labour + fees),
        "mentions": {"mot_devis_ou_ordre_reparation": True, "duree_validite_indiquee": True,
                     "zone_bon_pour_accord": True, "taux_horaire_affiche": True},
    }
    return ExtractionPage.model_validate(data).model_dump(mode="json")


CSS = """
* { font-family: sans-serif; font-size: 9pt; }
h1 { font-size: 15pt; margin: 0; }
h2 { font-size: 13pt; margin: 10pt 0 2pt 0; }
h3 { font-size: 10pt; margin: 8pt 0 2pt 0; }
p { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 0.5pt solid #555; padding: 2pt 3pt; }
th { background-color: #e6e6e6; text-align: left; }
td.num { text-align: right; }
.box { border: 0.8pt solid #000; padding: 6pt 6pt 40pt 6pt; margin-top: 10pt; }
.foot { font-size: 7pt; color: #555; margin-top: 12pt; }
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def row(cells: list, numeric_from: int) -> str:
    tds = "".join(f'<td class="num">{esc(c)}</td>' if i >= numeric_from else f"<td>{esc(c)}</td>"
                  for i, c in enumerate(cells))
    return f"<tr>{tds}</tr>"


def table(headers: list[str], rows: list[list], numeric_from: int) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(row(r, numeric_from) for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def quote_html(page: dict) -> str:
    g, v, h, t = page["emetteur"], page["vehicule"], page["en_tete"], page["totaux"]
    parts = [[p["reference"], p["designation"], p["quantite"], p["prix_unitaire_ht"], p["remise"] or "-",
              p["montant_ht"], p["taux_tva"]] for p in page["pieces"]]
    labour = [[m["designation"], m["categorie"], m["temps"], m["taux_horaire_ht"], m["montant_ht"]]
              for m in page["main_oeuvre"]]
    fees = [[f["designation"], f["montant_ht"]] for f in page["frais_divers"]]
    rates = " - ".join(f"{cat} : {fr(rate)} € HT" for cat, rate in HOURLY_RATES.items())
    return "".join([
        f"<h1>{esc(g['nom'])}</h1>",
        f"<p>{esc(g['adresse'])}<br/>{esc(g['code_postal'])} {esc(g['ville'])}<br/>"
        f"SIRET : {esc(g['siret'])} - Tél. 01 99 00 12 34</p>",
        f"<h2>DEVIS N° {esc(h['numero_document'])}</h2>",
        f"<p>Date : {esc(h['date_document'])} - Valable jusqu'au : {esc(h['date_fin_validite'])} "
        "(devis valable 30 jours)</p>",
        "<h3>Véhicule</h3>",
        table(["Immatriculation", "Marque / modèle", "VIN", "Kilométrage", "Mise en circulation"],
              [[v["immatriculation"], v["marque_modele"], v["vin"], v["kilometrage"],
                v["date_mise_circulation"]]], numeric_from=99),
        "<h3>Pièces</h3>",
        table(["Réf.", "Désignation", "Qté", "P.U. HT", "Remise", "Montant HT", "TVA"], parts, numeric_from=2),
        "<h3>Main-d'oeuvre</h3>",
        table(["Désignation", "Cat.", "Temps", "Taux horaire HT", "Montant HT"], labour, numeric_from=2),
        f"<p>Taux horaires : {esc(rates)}</p>",
        "<h3>Frais divers</h3>",
        table(["Désignation", "Montant HT"], fees, numeric_from=1),
        "<h3>Récapitulatif</h3>",
        table(["Total HT", f"TVA {t['taux_tva']}", "Total TTC"],
              [[t["total_ht"] + " €", t["total_tva"] + " €", t["total_ttc"] + " €"]], numeric_from=0),
        '<div class="box">Bon pour accord - date et signature du client :</div>',
        '<p class="foot">Document fictif généré pour les tests du projet Cloud_X_LLM. '
        "Garage, numéros et véhicule inventés, sans valeur commerciale.</p>",
    ])


def render_pdf(body_html: str) -> bytes:
    """One A4 page, rendered with the HTML engine of PyMuPDF, without dates or random IDs."""
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    story = pymupdf.Story(html=body_html, user_css=CSS)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (PAGE_MARGIN_PT, PAGE_MARGIN_PT, -PAGE_MARGIN_PT, -PAGE_MARGIN_PT)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    with pymupdf.open("pdf", buffer.getvalue()) as doc:
        doc.set_metadata(EMPTY_METADATA)
        return doc.tobytes(garbage=4, deflate=True, no_new_id=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Write the fictitious quote PDF and its expected extraction")
    p.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help=f"output PDF (default {DEFAULT_PDF})")
    args = p.parse_args()
    page = expected_page()
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.write_bytes(render_pdf(quote_html(page)))
    expected_path = args.pdf.with_suffix(".expected.json")
    expected_path.write_text(json.dumps([page], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.pdf} and {expected_path} written")


if __name__ == "__main__":
    main()
