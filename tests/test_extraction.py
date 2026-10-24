"""Unit tests for the vision extraction schema and the multi-page merge."""
import pytest
from pydantic import ValidationError

from src.schemas.extraction import (
    ExtractionPage,
    Lisibilite,
    ResultatCT,
    TypeDocument,
    merge_pages,
)

PIECE_FIELDS = ("reference", "quantite", "prix_unitaire_ht", "remise", "montant_ht", "taux_tva")
MENTION_FIELDS = ("mot_devis_ou_ordre_reparation", "duree_validite_indiquee",
                  "zone_bon_pour_accord", "taux_horaire_affiche")


def empty_page_data() -> dict:
    return {
        "type_document": "AUTRE",
        "lisibilite": "BONNE",
        "emetteur": dict.fromkeys(("nom", "adresse", "code_postal", "ville", "siret")),
        "vehicule": dict.fromkeys(("immatriculation", "marque_modele", "vin", "kilometrage",
                                   "date_mise_circulation")),
        "en_tete": dict.fromkeys(("numero_document", "date_document", "date_fin_validite")),
        "pieces": [],
        "main_oeuvre": [],
        "frais_divers": [],
        "resultat_ct": "NON_APPLICABLE",
        "defauts_ct": [],
        "totaux": dict.fromkeys(("total_ht", "taux_tva", "total_tva", "total_ttc")),
        "mentions": dict.fromkeys(MENTION_FIELDS, False),
    }


def piece(designation: str) -> dict:
    return {**dict.fromkeys(PIECE_FIELDS), "designation": designation}


def test_empty_page_is_valid():
    page = ExtractionPage.model_validate(empty_page_data())
    assert page.type_document is TypeDocument.AUTRE


def test_schema_forbids_extra_fields():
    data = empty_page_data()
    data["emetteur"]["telephone"] = "0102030405"
    with pytest.raises(ValidationError):
        ExtractionPage.model_validate(data)


def test_json_schema_is_closed():
    schema = ExtractionPage.model_json_schema()
    assert schema["additionalProperties"] is False


def test_pieces_max_length():
    data = empty_page_data()
    data["pieces"] = [piece(f"Piece {i}") for i in range(41)]
    with pytest.raises(ValidationError):
        ExtractionPage.model_validate(data)


def test_merge_pages_empty_raises():
    with pytest.raises(ValueError):
        merge_pages([])


def test_merge_single_page_is_identity():
    page = ExtractionPage.model_validate(empty_page_data())
    assert merge_pages([page]) is page


def make_two_pages() -> list[ExtractionPage]:
    first, last = empty_page_data(), empty_page_data()
    first["type_document"] = "DEVIS_GARAGE"
    first["emetteur"]["nom"] = "Garage Martin"
    first["mentions"]["zone_bon_pour_accord"] = True
    first["pieces"] = [piece(f"Filtre {i}") for i in range(30)]
    first["defauts_ct"] = [{"code": " 1.2.3 ", "libelle": "Feu", "gravite": "MAJEURE"}]
    first["totaux"]["total_ht"] = "80,00"
    first["totaux"]["total_ttc"] = "100,00"

    last["lisibilite"] = "MAUVAISE"
    last["emetteur"]["nom"] = "Autre nom"
    last["resultat_ct"] = "DEFAVORABLE_MAJEURE"
    last["pieces"] = [piece(f"Plaquette {i}") for i in range(30)]
    last["defauts_ct"] = [{"code": "1.2.3", "libelle": "Feu", "gravite": "MAJEURE"}]
    last["totaux"]["total_ttc"] = "120,00"
    return [ExtractionPage.model_validate(first), ExtractionPage.model_validate(last)]


def test_merge_two_pages():
    merged = merge_pages(make_two_pages())
    assert merged.type_document is TypeDocument.DEVIS_GARAGE  # AUTRE pages are ignored
    assert merged.lisibilite is Lisibilite.MAUVAISE           # worst page wins
    assert merged.emetteur.nom == "Garage Martin"             # first non-empty value
    assert merged.mentions.zone_bon_pour_accord is True       # True on any page
    assert merged.resultat_ct is ResultatCT.DEFAVORABLE_MAJEURE
    assert len(merged.defauts_ct) == 1                        # same code and severity
    assert merged.totaux.total_ttc == "120,00"                # last page first
    assert merged.totaux.total_ht == "80,00"                  # fallback to earlier page


def test_merged_lists_may_exceed_page_limit():
    merged = merge_pages(make_two_pages())
    assert len(merged.pieces) == 60
