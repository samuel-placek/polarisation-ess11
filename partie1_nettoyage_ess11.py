"""
Partie 1 – Récupération, nettoyage et enrichissement des données ESS11.

Ce script part du fichier brut ESS11 (https://ess.sikt.no/en/datafile/242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef), 
enrichie de certaines bases de données et produit :

1. une heatmap des valeurs manquantes par pays et par variable : cela permet de faire le tri des variables ou des pays à exclure
2. on retravaille : codes de non-réponse -> NaN, recodage dans un sens commun,
   normalisation sur [0, 1], renommage lisible, conversion des types ;
3. on crée de nouvelles variables politiques `lrscale`, `lrscaleabs` et `politiscale` ;
4. un enrichissement : partis votés (PartyFacts + ParlGov) et données pays (API Eurostat) ;
5. un fichier propre (`ess11_propre` ou `.csv`) et un dictionnaire des variables.


Fichiers de données (dans `data/`, à côté de ce script)
-------------------------------------------------------
- `data/ESS11e04_2.csv` : fichier intégré ESS11, édition 4.2 (portail ESS / Sikt) ;
- `data/externes/essprt-all.csv` : table PartyFacts des partis de l'ESS
  (https://github.com/hdigital/partyfactsdata/blob/main/import/essprtv/essprt-all.csv) ;
- `data/externes/partyfacts-external-parties.csv` : table de liens de PartyFacts entre toutes
  les bases de partis (page Download du site PartyFacts, fichier « external parties ») ;
- `data/externes/view_party.csv` : table des partis de ParlGov
  (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2VZ5ZC).

Sur Onyxia (SSP Cloud), les fichiers absents de `data/` sont copiés automatiquement depuis
le bucket S3 `BUCKET`. Ailleurs, il suffit de les placer dans `data/` à la main.
"""

#Importation des modules et gestions des données pour une utillisation depuis onyxia


from pathlib import Path
import os
import re
import subprocess
import sys
import warnings

# Installation des packages qui ne sont pas fournis par défaut sur Onyxia
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "eurostat", "s3fs"])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # pas d'écran : les figures sont seulement enregistrées
import matplotlib.pyplot as plt
import seaborn as sns
import eurostat
import s3fs


RACINE = Path(__file__).resolve().parent
CHEMIN_ESS = RACINE / "data" / "ESS11e04_2.csv"
DOSSIER_EXTERNES = RACINE / "data" / "externes"
DOSSIER_SORTIE = RACINE / "output"

# Bucket S3 d'Onyxia : chemin dans le bucket -> chemin local attendu par le script
BUCKET = "gleguestensae"
FICHIERS_S3 = {
    "ESS11e04_2.csv": CHEMIN_ESS,
    "essprt-all.csv": DOSSIER_EXTERNES / "essprt-all.csv",
    "partyfacts-external-parties.csv": DOSSIER_EXTERNES / "partyfacts-external-parties.csv",
    "view_party.csv": DOSSIER_EXTERNES / "view_party.csv",
}

def recuperer_donnees_s3():
    """Copie depuis le bucket S3 vers data/ les fichiers absents en local."""

    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]})
    print("[S3] Contenu du bucket :", fs.ls(BUCKET))
    for cle, local in FICHIERS_S3.items():
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            fs.get(f"{BUCKET}/{cle}", str(local))
            print(f"[S3] Téléchargé : {cle} -> {local}")
        except FileNotFoundError:
            print(f"[S3] Absent du bucket : {BUCKET}/{cle}")


# Chargement des données

def charger_ess(chemin):
    """Charge le CSV ESS et met les noms de colonnes en minuscules."""
    df = pd.read_csv(chemin, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["cntry"] = df["cntry"].astype("string").str.upper().str.strip()
    print(f"{len(df):,} répondants, {df.shape[1]} colonnes, {df['cntry'].nunique()} pays")
    return df

"""
On retravaille chaque variable ESS :
- nom : nouveau nom lisible ;
- vmin, vmax : bornes valides de l'échelle. Tout ce qui est en dehors
(7/8/9, 77/88/99, 55, 6666...) correspond à un code de non-réponse et devient NaN ;
- inverser : True si l'échelle doit être retournée pour qu'une valeur haute ait le sens
indiqué en commentaire de chaque groupe ;
- groupe : groupe thématique ou bloc de variables.

Après recodage, une valeur haute signifie *plus ouvert / progressiste /
confiant / satisfait / religieux / en bonne santé*, selon le groupe."""

ANNEE_REFERENCE = 2023
ROUND_ESS = 11 

VARIABLES = {
    # --- Immigration (haut = favorable à l'immigration) ---
    "imsmetn": dict(nom="immig_meme_ethnie",  vmin=1, vmax=4,  inverser=True,  groupe="immigration"),
    "imdfetn": dict(nom="immig_autre_ethnie", vmin=1, vmax=4,  inverser=True,  groupe="immigration"),
    "impcntr": dict(nom="immig_pays_pauvres", vmin=1, vmax=4,  inverser=True,  groupe="immigration"),
    "imbgeco": dict(nom="immig_economie",     vmin=0, vmax=10, inverser=False, groupe="immigration"),
    "imueclt": dict(nom="immig_culture",      vmin=0, vmax=10, inverser=False, groupe="immigration"),
    "imwbcnt": dict(nom="immig_pays_meilleur",vmin=0, vmax=10, inverser=False, groupe="immigration"),
    # --- LGBT / mœurs (haut = plus libéral) ---
    "freehms":  dict(nom="lgbt_libre",          vmin=1, vmax=5, inverser=True,  groupe="lgbt_moeurs"),
    "hmsfmlsh": dict(nom="lgbt_pas_honte",      vmin=1, vmax=5, inverser=False, groupe="lgbt_moeurs"),
    "hmsacld":  dict(nom="lgbt_adoption",       vmin=1, vmax=5, inverser=True,  groupe="lgbt_moeurs"),
    # --- Redistribution (haut = favorable à la redistribution) ---
    "gincdif": dict(nom="redistrib_reduire_inegalites", vmin=1, vmax=5, inverser=True, groupe="redistribution"),
    # --- Écologie (haut = plus préoccupé / engagé) ---
    # ccnthum : 55 = « le climat ne change pas », traité comme manquant (hors bornes)
    "ccnthum": dict(nom="eco_cause_humaine",     vmin=1, vmax=5,  inverser=False, groupe="ecologie"),
    "ccrdprs": dict(nom="eco_responsabilite",    vmin=0, vmax=10, inverser=False, groupe="ecologie"),
    "wrclmch": dict(nom="eco_inquietude",        vmin=1, vmax=5,  inverser=False, groupe="ecologie"),
    # --- Confiance institutionnelle (haut = confiance) ---
    "trstprl": dict(nom="confiance_parlement",   vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstlgl": dict(nom="confiance_justice",     vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstplc": dict(nom="confiance_police",      vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstplt": dict(nom="confiance_politiciens", vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstprt": dict(nom="confiance_partis",      vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstep":  dict(nom="confiance_parl_europeen", vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstun":  dict(nom="confiance_onu",         vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    # --- Europe (haut = pour plus d'intégration) ---
    "euftf": dict(nom="europe_integration", vmin=0, vmax=10, inverser=False, groupe="europe"),
    # --- Satisfaction (haut = satisfait) ---
    "stfeco": dict(nom="satisf_economie",    vmin=0, vmax=10, inverser=False, groupe="satisfaction"),
    "stfgov": dict(nom="satisf_gouvernement",vmin=0, vmax=10, inverser=False, groupe="satisfaction"),
    "stfdem": dict(nom="satisf_democratie",  vmin=0, vmax=10, inverser=False, groupe="satisfaction"),
    # --- Religiosité (haut = plus religieux) ---
    "rlgdgr": dict(nom="relig_degre",       vmin=0, vmax=10, inverser=False, groupe="religiosite"),
    "rlgatnd":dict(nom="relig_assiduite",   vmin=1, vmax=7,  inverser=True,  groupe="religiosite"),
    "pray":   dict(nom="relig_priere",      vmin=1, vmax=7,  inverser=True,  groupe="religiosite"),
    # --- Santé (haut = meilleure santé physique et mentale) ---
    "health":  dict(nom="sante_auto_evaluee", vmin=1, vmax=5, inverser=True,  groupe="sante"),
    "hlthhmp": dict(nom="sante_pas_genee",    vmin=1, vmax=3, inverser=False, groupe="sante"),
    # échelle de dépression CES-D 8 (module « Social inequalities in health »), 1 = jamais ... 4 = tout le temps
    "fltdpr":  dict(nom="sante_pas_deprime",     vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "flteeff": dict(nom="sante_pas_effort",      vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "slprl":   dict(nom="sante_sommeil_calme",   vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "wrhpp":   dict(nom="sante_heureux",         vmin=1, vmax=4, inverser=False, groupe="sante"),
    "fltlnl":  dict(nom="sante_pas_seul",        vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "enjlf":   dict(nom="sante_profite_vie",     vmin=1, vmax=4, inverser=False, groupe="sante"),
    "fltsd":   dict(nom="sante_pas_triste",      vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "cldgng":  dict(nom="sante_motive",          vmin=1, vmax=4, inverser=True,  groupe="sante"),
    # --- Genre (module « Gender in contemporary Europe ») : haut = plus égalitaire ---
    "eqwrkbg": dict(nom="genre_emploi_egal_famille",  vmin=0, vmax=6, inverser=False, groupe="genre"),
    "eqmgmbg": dict(nom="genre_direction_egale",      vmin=0, vmax=6, inverser=False, groupe="genre"),
    "eqpaybg": dict(nom="genre_salaire_egal_eco",     vmin=0, vmax=6, inverser=False, groupe="genre"),
    "eqparep": dict(nom="genre_parite_parlement",     vmin=1, vmax=5, inverser=True,  groupe="genre"),
    "eqparlv": dict(nom="genre_conge_parental_egal",  vmin=1, vmax=5, inverser=True,  groupe="genre"),
    "freinsw": dict(nom="genre_licencier_insultes",   vmin=1, vmax=5, inverser=True,  groupe="genre"),
    "fineqpy": dict(nom="genre_amende_inegalite",     vmin=1, vmax=5, inverser=True,  groupe="genre"),
    "wsekpwr": dict(nom="genre_rejet_femmes_pouvoir", vmin=1, vmax=5, inverser=True,  groupe="genre"),
    "weasoff": dict(nom="genre_rejet_femmes_offensees", vmin=1, vmax=5, inverser=True, groupe="genre"),
    "wexashr": dict(nom="genre_rejet_harcelement_exagere", vmin=1, vmax=5, inverser=True, groupe="genre"),
    "wlespdm": dict(nom="genre_percoit_ecart_salaire", vmin=1, vmax=5, inverser=False, groupe="genre"),
    "wprtbym": dict(nom="genre_rejet_protection",     vmin=1, vmax=5, inverser=False, groupe="genre"),
    # wbrgwrm (sexisme « bienveillant ») n'est pas corrélé au reste du groupe (r = 0,01 avec
    # l'indice) : on garde la variable mais on la sort de l'indice genre.
    "wbrgwrm": dict(nom="genre_rejet_femmes_morales", vmin=1, vmax=5, inverser=False, groupe="genre_hors_indice"),

    # ===== Variables politiques (hors groupes thématiques) =====
    "lrscale": dict(nom="lrscale", vmin=0, vmax=10, inverser=False, groupe="politique"),
    "polintr": dict(nom="pol_interet", vmin=1, vmax=4, inverser=True, groupe="politisation_interet"),
    # actions politiques : 1 = oui, 2 = non → après inversion et normalisation, oui = 1
    "contplt":  dict(nom="pol_contact_elu",       vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "donprty":  dict(nom="pol_don_parti",         vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "badge":    dict(nom="pol_badge",             vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "sgnptit":  dict(nom="pol_petition",          vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "pbldmna":  dict(nom="pol_manifestation",     vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "bctprd":   dict(nom="pol_boycott",           vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "pstplonl": dict(nom="pol_post_en_ligne",     vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "volunfp":  dict(nom="pol_benevolat",         vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    # vote : 3 = « pas en âge / pas éligible » → hors bornes → NaN
    "vote":     dict(nom="pol_a_vote",            vmin=1, vmax=2, inverser=True, groupe="politisation_partisan"),
    "clsprty":  dict(nom="pol_proche_parti",      vmin=1, vmax=2, inverser=True, groupe="politisation_partisan"),
}

GROUPES_THEMATIQUES = [
    "immigration", "lgbt_moeurs", "redistribution", "ecologie",
    "confiance_institutionnelle", "europe", "satisfaction", "religiosite",
    "genre", "sante",
]

# Variables sociodémographiques : nom ESS -> (nouveau nom, bornes valides)
SOCIODEMO = {
    "agea":     ("age",15, 110),
    "yrbrn":    ("annee_naissance",1900,2012),
    "gndr":     ("genre_resp",1,2),     # 1 = homme, 2 = femme
    "eisced":   ("education_isced",1,7),     # 0 et 55 = non classable → NaN
    "eduyrs":   ("annees_etudes",0,40),
    "hinctnta": ("revenu_decile",1,10),
    "domicil":  ("lieu_vie",1,5),     # 1 = grande ville ... 5 = ferme isolée
}

POIDS = ["anweight", "pspwght", "pweight", "dweight"]
IDENTIFIANTS = ["idno", "cntry", "essround"]

# Codes de non-réponse (NaN)


# Les codes de non-réponse de l'ESS changent selon l'échelle (7/8/9 pour une échelle en 1-4,
# 77/88/99 pour une échelle en 0-10, etc.). On garde donc, variable par variable, uniquement
# les valeurs comprises dans les bornes.

# Pour `lrscale`, on garde aussi l'information « ne sait pas » (code 88) dans une indicatrice
# `lrscale_nsp` : ne pas savoir se placer est en soi un signe de faible politisation.

def colonnes_presentes(df, liste):
    presentes = [c for c in liste if c in df.columns]
    absentes = [c for c in liste if c not in df.columns]
    if absentes:
        warnings.warn(f"Variables absentes du fichier, ignorées : {absentes}")
    return presentes


def codes_manquants_vers_nan(df):
    """Remplace par NaN toute valeur hors des bornes valides (VARIABLES et SOCIODEMO)."""
    df = df.copy()
    bornes = {k: (v["vmin"], v["vmax"]) for k, v in VARIABLES.items()}
    bornes.update({k: (vmin, vmax) for k, (_, vmin, vmax) in SOCIODEMO.items()})

    if "lrscale" in df.columns:
        lr_brut = pd.to_numeric(df["lrscale"], errors="coerce")
        df["lrscale_nsp"] = (lr_brut == 88).astype("Int8")

    for col in colonnes_presentes(df, list(bornes)):
        vmin, vmax = bornes[col]
        x = pd.to_numeric(df[col], errors="coerce")
        df[col] = x.where(x.between(vmin, vmax))
    return df



# 1. Heatmap des valeurs manquantes par pays
# Une case = part de répondants d'un pays sans réponse valide à une variable.
# Une colonne entièrement foncée pour un pays signifie le plus souvent que la question n'a
# pas été posée dans ce pays.

def taux_manquants_par_pays(df,variables):
    variables = colonnes_presentes(df, variables)
    taux = df[variables].isna().groupby(df["cntry"]).mean()
    taux["MOYENNE"] = taux.mean(axis=1)
    return taux.sort_values("MOYENNE", ascending=False)


def heatmap_manquants(taux,chemin_png):
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * taux.shape[1]), max(6, 0.35 * taux.shape[0])))
    sns.heatmap(taux, cmap="Reds", vmin=0, vmax=1, cbar_kws={"label": "Part de valeurs manquantes"},
                linewidths=0.3, linecolor="white", ax=ax)
    ax.set_xlabel("Variable ESS")
    ax.set_ylabel("Pays (trié par taux moyen de manquants)")
    ax.set_title("ESS11 : valeurs manquantes par pays et par variable")
    plt.tight_layout()
    if chemin_png:
        fig.savefig(chemin_png, dpi=150)
        print(f"Heatmap enregistrée : {chemin_png}")
    return fig

# Exclusion de pays / variables
# =============================================================================
# Après analyse de la heatmap, il apparaît que tous les pays affichent pour la plupart des variables des taux de 
#réponse satisfaisants, à l'exception de l'Ukraine.

PAYS_EXCLUS = ["UA"]


def exclure_pays(df, pays_exclus):
    """Retire de la base les répondants des pays listés (codes ISO à 2 lettres)."""
    n_avant = len(df)
    df = df[~df["cntry"].isin(pays_exclus)].copy()
    print(f"Pays exclus : {pays_exclus} -> {n_avant - len(df):,} répondants retirés, "
          f"{len(df):,} restants")
    return df



# 2. Recodage, normalisation, renommage


def recoder_et_normaliser(df):
    """Crée, pour chaque variable de VARIABLES, une colonne au nom lisible, normalisée sur
    [0, 1] et dans le sens défini plus haut. `lrscale` reste sur 0-10 pour rester lisible."""
    df = df.copy()
    for col in colonnes_presentes(df, list(VARIABLES)):
        p = VARIABLES[col]
        x = df[col].astype("Float64")
        if p["inverser"]:
            x = p["vmin"] + p["vmax"] - x
        if col == "lrscale":
            df["lrscale"] = x
        else:
            df[p["nom"]] = (x - p["vmin"]) / (p["vmax"] - p["vmin"])
    return df


def nettoyer_sociodemo(df):
    df = df.copy()
    presentes = colonnes_presentes(df, list(SOCIODEMO))
    df = df.rename(columns={k: SOCIODEMO[k][0] for k in presentes})

    if "genre_resp" in df:
        df["femme"] = (df["genre_resp"] == 2).astype("Int8").where(df["genre_resp"].notna())
    if "age" in df:
        df["tranche_age"] = pd.cut(df["age"], bins=[14, 29, 44, 59, 74, 120],
                                   labels=["15-29", "30-44", "45-59", "60-74", "75+"])
    if "annee_naissance" in df:
        decennie = (df["annee_naissance"] // 10 * 10).astype("Int64")
        df["cohorte"] = decennie.astype("string") + "s"
    if "lieu_vie" in df:
        df["urbain"] = (df["lieu_vie"] <= 2).astype("Int8").where(df["lieu_vie"].notna())
    if "cntry" in df:
        df["cntry"] = df["cntry"].astype("category")
    return df



# 3. Variables politiques : `lrscale`, `lrscaleabs`, `politiscale`

#
# - `lrscale` : auto-positionnement gauche (0) / droite (10) ;
# - `lrscaleabs` = |lrscale − 5| : extrémité idéologique, de 0 (centre) à 5 (extrême) ;
# - `politiscale` : indice de politisation sur [0, 1], moyenne de trois composantes :
#   - intérêt pour la politique (`polintr`) ;
#   - participation : part des 8 actions politiques réalisées dans les 12 derniers mois ;
#   - lien partisan : a voté aux dernières élections, se sent proche d'un parti.
#   L'indice est calculé si au moins 2 des 3 composantes sont disponibles.

def noms_du_groupe(groupe, df):
    return [p["nom"] for k, p in VARIABLES.items() if p["groupe"] == groupe and p["nom"] in df.columns]


def moyenne_lignes(df, cols, min_part=0.5):
    """Moyenne par ligne, NaN si moins de `min_part` des colonnes sont renseignées."""
    if not cols:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    bloc = df[cols].astype("Float64")
    n_ok = bloc.notna().sum(axis=1)
    return bloc.mean(axis=1).where(n_ok >= np.ceil(min_part * len(cols)))


def creer_variables_politiques(df):
    df = df.copy()
    if "lrscale" in df:
        df["lrscaleabs"] = (df["lrscale"] - 5).abs()

    composantes = pd.DataFrame({
        "politi_interet": moyenne_lignes(df, noms_du_groupe("politisation_interet", df)),
        "politi_participation": moyenne_lignes(df, noms_du_groupe("politisation_participation", df)),
        "politi_partisan": moyenne_lignes(df, noms_du_groupe("politisation_partisan", df)),
    })
    df = pd.concat([df, composantes], axis=1)
    df["politiscale"] = moyenne_lignes(df, list(composantes.columns), min_part=2 / 3)
    return df

def creer_indices_thematiques(df):
    df = df.copy()
    for g in GROUPES_THEMATIQUES:
        cols = noms_du_groupe(g, df)
        if not cols:
            warnings.warn(f"Groupe « {g} » : aucune variable disponible, indice non créé.")
            continue
        df[f"idx_{g}"] = moyenne_lignes(df, cols)
    return df

# 4. Enrichissement 1 : partis votés (PartyFacts + ParlGov)

# Le vote est stocké dans une variable par pays (`prtvtfat` pour l'Autriche, `prtvede2` pour
# l'Allemagne, etc.), avec des codes propres à chaque pays. PartyFacts identifie chaque parti
# de l'ESS par un `ess_id` de la forme `PAYS-ROUND-CODE-v`, par exemple `FR-11-3-v`.
# En Allemagne et en Lituanie, le tour de scrutin est ajouté : `DE-11-2-v-de2`.

# Chaîne de jointures :
# `ess_id` → identifiant PartyFacts → identifiant ParlGov → famille de partis et position
# gauche/droite du parti selon les experts (0-10), avec les axes État/marché,
# liberté/autorité et UE.

# Pour les pays avec plusieurs variables de vote, la variable à retenir (À VÉRIFIER dans le codebook)
TOUR_RETENU = {"de": "2", "lt": "1"}   # DE : prtvgde2 (2e voix, liste) ; LT : prtvclt1 (1er vote, parti)


def construire_ess_id(df):
    cols_vote = [c for c in df.columns if re.fullmatch(r"prtv[a-z]*[a-z]{2}\d?", c)]
    ess_id = pd.Series(pd.NA, index=df.index, dtype="string")
    cntry = df["cntry"].astype("string").str.lower()

    for col in cols_vote:
        tour = col[-1] if col[-1].isdigit() else ""
        pays = col[-3:-1] if tour else col[-2:]
        if tour and TOUR_RETENU.get(pays) != tour:
            continue
        code = pd.to_numeric(df[col], errors="coerce")
        masque = (cntry == pays) & code.between(1, 65)   # 66/77/88/99 = sans objet / NSP / refus
        suffixe = f"-{pays}{tour}" if tour else ""
        ess_id[masque] = (pays.upper() + f"-{ROUND_ESS}-" +
                          code[masque].astype("Int64").astype("string") + "-v" + suffixe)
    print(f"Vote identifié pour {ess_id.notna().mean():.1%} des répondants "
          f"({len(cols_vote)} variables prtv* trouvées)")
    return ess_id


def lire_csv_externe(nom, colonnes_requises):
    chemin = DOSSIER_EXTERNES / nom
    if not chemin.exists():
        print(f"[PartyFacts] Fichier manquant : {chemin} → étape sautée.")
        return None
    t = pd.read_csv(chemin, low_memory=False)
    manquantes = set(colonnes_requises) - set(t.columns)
    if manquantes:
        raise KeyError(f"{nom} : colonnes {manquantes} introuvables. Colonnes disponibles : {list(t.columns)}")
    return t


def enrichir_partis(df):
    df = df.copy()
    df["parti_ess_id"] = construire_ess_id(df)

    essprt = lire_csv_externe("essprt-all.csv", ["ess_id"])
    liens = lire_csv_externe("partyfacts-external-parties.csv",
                             ["dataset_key", "dataset_party_id", "partyfacts_id"])
    parlgov = lire_csv_externe("view_party.csv", ["party_id", "family_name", "left_right"])
    if essprt is None or liens is None or parlgov is None:
        return df

    liens["dataset_party_id"] = liens["dataset_party_id"].astype("string")

    # a) ess_id -> partyfacts_id (directement si la colonne existe, sinon via first_ess_id)
    if "partyfacts_id" in essprt.columns:
        pf = essprt[["ess_id", "partyfacts_id"]]
    else:
        pf_ess = liens.loc[liens["dataset_key"] == "essprtv", ["dataset_party_id", "partyfacts_id"]]
        pf = essprt[["ess_id", "first_ess_id"]].astype("string").merge(
            pf_ess, left_on="first_ess_id", right_on="dataset_party_id", how="left"
        )[["ess_id", "partyfacts_id"]]
    pf = pf.dropna().drop_duplicates("ess_id").astype({"ess_id": "string", "partyfacts_id": "Int64"})

    # b) partyfacts_id -> party_id ParlGov
    pg_liens = (liens.loc[liens["dataset_key"] == "parlgov", ["partyfacts_id", "dataset_party_id"]]
                .rename(columns={"dataset_party_id": "parlgov_id"})
                .drop_duplicates("partyfacts_id"))
    pg_liens["parlgov_id"] = pd.to_numeric(pg_liens["parlgov_id"], errors="coerce").astype("Int64")
    pg_liens["partyfacts_id"] = pg_liens["partyfacts_id"].astype("Int64")

    # c) caractéristiques ParlGov
    cols_pg = {"party_id": "parlgov_id", "family_name": "parti_famille", "left_right": "parti_lr",
               "state_market": "parti_etat_marche", "liberty_authority": "parti_liberte_autorite",
               "eu_anti_pro": "parti_pro_ue"}
    pg = parlgov[[c for c in cols_pg if c in parlgov.columns]].rename(columns=cols_pg)
    pg["parlgov_id"] = pg["parlgov_id"].astype("Int64")

    table = pf.merge(pg_liens, on="partyfacts_id", how="left").merge(pg, on="parlgov_id", how="left")
    table = table.rename(columns={"ess_id": "parti_ess_id", "partyfacts_id": "parti_partyfacts_id"})

    n_avant = len(df)
    df = df.merge(table, on="parti_ess_id", how="left")
    assert len(df) == n_avant, "La jointure a dupliqué des lignes"
    a_vote = df["parti_ess_id"].notna()
    print(f"[PartyFacts] Parti relié à ParlGov pour {df.loc[a_vote, 'parti_lr'].notna().mean():.1%} "
          f"des votants identifiés")
    return df



# Enrichissement 2 : données pays via l'API Eurostat
#
# Indicateurs pris pour l'année du terrain (2023, ou l'année d'avant si 2023 manque) :
# - PIB par habitant en standard de pouvoir d'achat (indice UE27 = 100) : `tec00114` ;
# - coefficient de Gini du revenu disponible : `ilc_di12` ;
# - taux de chômage des 15-74 ans : `une_rt_a`.
#
# Si un jeu de données a plusieurs dimensions (unité, sexe...), il faut les fixer dans
# `filtres` ; sinon la fonction s'arrête et affiche les valeurs possibles.
# Eurostat ne couvre pas Israël, couvre peu ou pas le Royaume-Uni depuis 2020, et seulement
# en partie les pays candidats : leurs valeurs resteront NaN.

INDICATEURS_EUROSTAT = {
    "pib_hab_spa":   dict(code="tec00114", filtres={}),
    "gini":          dict(code="ilc_di12", filtres={}),
    "taux_chomage":  dict(code="une_rt_a", filtres={"age": "Y15-74", "sex": "T", "unit": "PC_ACT"}),
}
ESS_VERS_EUROSTAT = {"GR": "EL", "GB": "UK"}   # codes pays qui diffèrent


def indicateur_eurostat(code, filtres, annee=ANNEE_REFERENCE):
    import eurostat
    brut = eurostat.get_data_df(code)
    col_geo = next(c for c in brut.columns if "geo" in c.lower())
    brut = brut.rename(columns={col_geo: "geo"})
    for dim, val in filtres.items():
        brut = brut[brut[dim] == val]

    annees = [c for c in brut.columns if re.fullmatch(r"\d{4}", str(c))]
    dims = [c for c in brut.columns if c not in annees + ["geo", "freq"]]
    if brut.duplicated("geo").any():
        valeurs = {d: brut[d].unique().tolist()[:10] for d in dims if brut[d].nunique() > 1}
        raise ValueError(f"{code} : plusieurs lignes par pays, fixez ces dimensions dans `filtres` : {valeurs}")

    serie = brut.set_index("geo")[str(annee)] if str(annee) in brut else pd.Series(dtype=float)
    if str(annee - 1) in brut:
        serie = serie.fillna(brut.set_index("geo")[str(annee - 1)])
    return pd.to_numeric(serie, errors="coerce")


def enrichir_eurostat(df):
    try:
        import eurostat  # noqa: F401
    except ImportError:
        print("[Eurostat] Package `eurostat` non installé (pip install eurostat) → étape sautée.")
        return df

    pays = pd.DataFrame({"cntry": df["cntry"].astype("string").unique()})
    pays["geo"] = pays["cntry"].replace(ESS_VERS_EUROSTAT)
    for nom, p in INDICATEURS_EUROSTAT.items():
        try:
            pays[nom] = pays["geo"].map(indicateur_eurostat(p["code"], p["filtres"]))
        except Exception as e:  # API indisponible, dimension à préciser...
            print(f"[Eurostat] {nom} ({p['code']}) non récupéré : {e}")
    print("[Eurostat] Pays sans données :",
          pays.loc[pays.drop(columns=["cntry", "geo"]).isna().all(axis=1), "cntry"].tolist())

    pays = pays.drop(columns="geo")
    pays.to_csv(DOSSIER_SORTIE / "indicateurs_pays_eurostat.csv", index=False)
    df = df.copy()
    df["cntry"] = df["cntry"].astype("string")
    return df.merge(pays, on="cntry", how="left").astype({"cntry": "category"})



# Pondérations

def moyenne_ponderee(x,poids):
    ok = x.notna() & poids.notna()
    return float(np.average(x[ok].astype(float), weights=poids[ok].astype(float)))



# Dictionnaire des variables et export

def dictionnaire_variables(df):
    lignes = []
    for k, p in VARIABLES.items():
        if p["nom"] in df.columns:
            echelle = "0-10" if k == "lrscale" else "0-1"
            lignes.append(dict(variable=p["nom"], source_ess=k, groupe=p["groupe"],
                               echelle=echelle, inversee=p["inverser"]))
    derivees = {
        "lrscale_nsp": "1 si « ne sait pas » à lrscale",
        "lrscaleabs": "|lrscale - 5|, extrémité idéologique (0-5)",
        "politi_interet": "composante intérêt (0-1)",
        "politi_participation": "part des actions politiques réalisées (0-1)",
        "politi_partisan": "a voté / proche d'un parti (0-1)",
        "politiscale": "indice de politisation (0-1)",
        "parti_lr": "position gauche-droite du parti voté (ParlGov, 0-10)",
        "parti_famille": "famille du parti voté (ParlGov)",
    }
    derivees.update({f"idx_{g}": f"indice thématique {g} (0-1)" for g in GROUPES_THEMATIQUES})
    for v, d in derivees.items():
        if v in df.columns:
            lignes.append(dict(variable=v, source_ess="dérivée", groupe="", echelle=d, inversee=""))
    return pd.DataFrame(lignes)


def exporter(df, nom="ess11_propre"):
    chemin = DOSSIER_SORTIE / f"{nom}.csv"
    df.to_csv(chemin, index=False)
    print(f"Export : {chemin}")


# Programme principal

def selection_colonnes_finales(df):
    colonnes = (
        colonnes_presentes(df, IDENTIFIANTS + POIDS)
        + [c for c in df.columns if c in {s[0] for s in SOCIODEMO.values()}
           or c in {"femme", "tranche_age", "cohorte", "urbain"}]
        + ["lrscale", "lrscale_nsp", "lrscaleabs", "politi_interet", "politi_participation",
           "politi_partisan", "politiscale"]
        + [p["nom"] for k, p in VARIABLES.items() if k != "lrscale" and p["nom"] in df.columns]
        + [c for c in df.columns if c.startswith("idx_")]
        + [c for c in df.columns if c.startswith("parti_")]
        + [c for c in INDICATEURS_EUROSTAT if c in df.columns]
    )
    return list(dict.fromkeys(c for c in colonnes if c in df.columns))


def main():
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    recuperer_donnees_s3()

    # Chargement et codes manquants
    brut = charger_ess(CHEMIN_ESS)
    df = codes_manquants_vers_nan(brut)

    # Heatmap des manquants
    taux = taux_manquants_par_pays(df, list(VARIABLES) + list(SOCIODEMO))
    print("\nTaux moyen de valeurs manquantes par pays :")
    print(taux["MOYENNE"].round(3).to_string())
    taux.to_csv(DOSSIER_SORTIE / "taux_manquants_par_pays.csv")
    fig = heatmap_manquants(taux, DOSSIER_SORTIE / "heatmap_manquants.png")
    plt.close(fig)

    df = exclure_pays(df, PAYS_EXCLUS)

    # Nettoyage et variables créées
    df = recoder_et_normaliser(df)
    df = nettoyer_sociodemo(df)
    df = creer_variables_politiques(df)
    df = creer_indices_thematiques(df)

    # Enrichissement
    df = enrichir_partis(df)
    df = enrichir_eurostat(df)

    # Export
    propre = df[selection_colonnes_finales(df)]
    exporter(propre)
    dictionnaire_variables(propre).to_csv(DOSSIER_SORTIE / "dictionnaire_variables.csv", index=False)
    print(f"\nBase finale : {propre.shape[0]:,} répondants, {propre.shape[1]} variables")
    return propre


if __name__ == "__main__":
    main()