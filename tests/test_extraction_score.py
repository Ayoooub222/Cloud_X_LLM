"""Unit tests for the field-level accuracy of vision extractions."""
from src.bench.extraction_score import flatten, normalize, score

EXPECTED = {
    "emetteur": {"nom": "Garage des Essais", "siret": "000 000 000 00000"},
    "pieces": [
        {"designation": "Filtre à huile", "montant_ht": "12,40", "remise": None},
        {"designation": "Filtre à air", "montant_ht": "18,75", "remise": "5 %"},
    ],
    "main_oeuvre": [{"temps": "1,5 h"}],
    "mentions": {"zone_bon_pour_accord": True},
}


def test_flatten_keeps_list_indexes():
    flat = flatten(EXPECTED)
    assert flat["pieces[1].montant_ht"] == "18,75"
    assert flat["mentions.zone_bon_pour_accord"] is True


def test_identical_extraction_scores_one():
    assert score(EXPECTED, EXPECTED)["field_accuracy"] == 1.0


def test_formatting_differences_still_match():
    predicted = {
        "emetteur": {"nom": "  garage des essais ", "siret": "00000000000000"},
        "pieces": [
            {"designation": "Filtre à huile", "montant_ht": "12.40 €", "remise": "-"},
            {"designation": "Filtre à air", "montant_ht": "18,75", "remise": "5%"},
        ],
        "main_oeuvre": [{"temps": "1h30"}],
        "mentions": {"zone_bon_pour_accord": True},
    }
    assert score(EXPECTED, predicted)["field_accuracy"] == 1.0


def test_missing_and_invented_lines_lower_the_score():
    predicted = {**EXPECTED, "pieces": EXPECTED["pieces"][:1] + [{"designation": "Invented"}]}
    result = score(EXPECTED, predicted)
    assert result["fields_matched"] < result["fields_total"]
    assert result["field_accuracy"] < 1.0


def test_normalize_keeps_plates_as_text():
    assert normalize("immatriculation", "AA-123-AA") == "aa-123-aa"
