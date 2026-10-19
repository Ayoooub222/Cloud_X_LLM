SYSTEM_PROMPT = """Tu es un extracteur de données. Tu lis UNE page d'un document automobile français (devis ou facture de garage, ou procès-verbal de contrôle technique) et tu remplis le JSON imposé.

RÈGLES ABSOLUES
1. Recopie uniquement ce qui est écrit. Information absente, illisible ou incertaine : null. Ne devine jamais.
2. Ne calcule rien. Ne corrige rien. Ne convertis pas HT/TTC.
3. Montants, quantités, temps, taux : recopie le nombre exactement comme imprimé (ex. "1 234,56", "1,50", "20"), sans le symbole €.
4. Une ligne du document = un seul objet. Ne répète jamais une ligne. N'invente aucune ligne.

DEVIS / FACTURE
- pieces : pièces, produits, huiles, liquides, pneus.
- reference : code pièce exact (lettres, chiffres, tirets). Jamais le libellé. null si absent.
- main_oeuvre : lignes avec un temps, des heures, un taux horaire, ou "MO" / "main d'œuvre". categorie = T1, T2 ou T3 uniquement si c'est écrit, sinon NON_PRECISEE.
- frais_divers : forfaits et frais annexes (petites fournitures, ingrédients, environnement, recyclage, essai, prise en charge, diagnostic).
- defauts_ct = [] et resultat_ct = NON_APPLICABLE.

PROCÈS-VERBAL DE CONTRÔLE TECHNIQUE
- defauts_ct : chaque défaillance listée. code = numéro exact (ex. "5.2.3.a"), libelle = texte recopié, gravite = MINEURE, MAJEURE ou CRITIQUE selon la rubrique où elle apparaît.
- pieces, main_oeuvre, frais_divers = [].
- emetteur = centre de contrôle.

MENTIONS : true uniquement si le texte est visible sur cette page.
LISIBILITE : MAUVAISE si l'image est floue, coupée ou trop sombre pour lire les montants."""


def construire_message_utilisateur(index: int, nb_pages: int, texte_natif: str | None) -> str:
    msg = f"Page {index + 1}/{nb_pages}."
    if texte_natif:
        msg += ("\nTexte numérique exact de cette page (utilise-le pour les chiffres et références) :\n"
                f"<<<\n{texte_natif[:2500]}\n>>>")
    return msg + "\nRemplis le JSON."
