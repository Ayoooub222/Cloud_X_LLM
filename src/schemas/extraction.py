from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")   # additionalProperties: false -> grammaire plus serrée


class TypeDocument(str, Enum):
    DEVIS_GARAGE = "DEVIS_GARAGE"
    FACTURE_GARAGE = "FACTURE_GARAGE"
    PV_CONTROLE_TECHNIQUE = "PV_CONTROLE_TECHNIQUE"
    AUTRE = "AUTRE"


class Lisibilite(str, Enum):
    BONNE = "BONNE"
    MOYENNE = "MOYENNE"
    MAUVAISE = "MAUVAISE"


class CategorieMO(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    NON_PRECISEE = "NON_PRECISEE"


class Gravite(str, Enum):
    MINEURE = "MINEURE"
    MAJEURE = "MAJEURE"
    CRITIQUE = "CRITIQUE"


class ResultatCT(str, Enum):
    FAVORABLE = "FAVORABLE"
    DEFAVORABLE_MAJEURE = "DEFAVORABLE_MAJEURE"
    DEFAVORABLE_CRITIQUE = "DEFAVORABLE_CRITIQUE"
    NON_APPLICABLE = "NON_APPLICABLE"


class Emetteur(_Strict):          # garage ou centre de contrôle technique
    nom: str | None
    adresse: str | None
    code_postal: str | None
    ville: str | None
    siret: str | None


class Vehicule(_Strict):
    immatriculation: str | None
    marque_modele: str | None
    vin: str | None
    kilometrage: str | None
    date_mise_circulation: str | None


class EnTete(_Strict):
    numero_document: str | None
    date_document: str | None
    date_fin_validite: str | None


class LignePiece(_Strict):
    reference: str | None
    designation: str
    quantite: str | None
    prix_unitaire_ht: str | None
    remise: str | None
    montant_ht: str | None
    taux_tva: str | None


class LigneMainOeuvre(_Strict):
    designation: str
    categorie: CategorieMO
    temps: str | None
    taux_horaire_ht: str | None
    montant_ht: str | None


class LigneFrais(_Strict):
    designation: str
    montant_ht: str | None


class DefautCT(_Strict):
    code: str
    libelle: str
    gravite: Gravite


class Totaux(_Strict):
    total_ht: str | None
    taux_tva: str | None
    total_tva: str | None
    total_ttc: str | None


class Mentions(_Strict):
    mot_devis_ou_ordre_reparation: bool
    duree_validite_indiquee: bool
    zone_bon_pour_accord: bool
    taux_horaire_affiche: bool


class ExtractionPage(_Strict):
    type_document: TypeDocument
    lisibilite: Lisibilite
    emetteur: Emetteur
    vehicule: Vehicule
    en_tete: EnTete
    pieces: list[LignePiece] = Field(..., max_length=40)
    main_oeuvre: list[LigneMainOeuvre] = Field(..., max_length=25)
    frais_divers: list[LigneFrais] = Field(..., max_length=15)
    resultat_ct: ResultatCT
    defauts_ct: list[DefautCT] = Field(..., max_length=40)
    totaux: Totaux
    mentions: Mentions


# ---------- Fusion multi-pages (déterministe) ----------
_RANG_LISIBILITE = {Lisibilite.BONNE: 0, Lisibilite.MOYENNE: 1, Lisibilite.MAUVAISE: 2}


def _fusion_champs(objs: list[BaseModel]) -> BaseModel:
    cls = type(objs[0])
    data = {}
    for nom in cls.model_fields:
        vals = [getattr(o, nom) for o in objs]
        data[nom] = any(vals) if all(isinstance(v, bool) for v in vals) \
            else next((v for v in vals if v not in (None, "")), None)
    return cls(**data)


def fusionner_pages(pages: list[ExtractionPage]) -> ExtractionPage:
    if not pages:
        raise ValueError("Aucune page extraite")
    if len(pages) == 1:
        return pages[0]
    types = [p.type_document for p in pages if p.type_document is not TypeDocument.AUTRE]
    resultats = [p.resultat_ct for p in pages if p.resultat_ct is not ResultatCT.NON_APPLICABLE]
    defauts, vus = [], set()
    for p in pages:
        for d in p.defauts_ct:
            cle = (d.code.strip(), d.gravite)
            if cle not in vus:
                vus.add(cle)
                defauts.append(d)
    # model_construct : la concaténation peut dépasser les maxItems par page
    return ExtractionPage.model_construct(
        type_document=Counter(types).most_common(1)[0][0] if types else TypeDocument.AUTRE,
        lisibilite=max((p.lisibilite for p in pages), key=_RANG_LISIBILITE.__getitem__),
        emetteur=_fusion_champs([p.emetteur for p in pages]),
        vehicule=_fusion_champs([p.vehicule for p in pages]),
        en_tete=_fusion_champs([p.en_tete for p in pages]),
        pieces=[l for p in pages for l in p.pieces],
        main_oeuvre=[l for p in pages for l in p.main_oeuvre],
        frais_divers=[l for p in pages for l in p.frais_divers],
        resultat_ct=resultats[0] if resultats else ResultatCT.NON_APPLICABLE,
        defauts_ct=defauts,
        totaux=_fusion_champs([p.totaux for p in reversed(pages)]),   # les totaux sont en dernière page
        mentions=_fusion_champs([p.mentions for p in pages]),
    )
