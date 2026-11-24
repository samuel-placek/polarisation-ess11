"""
Partie 1 - Récupération, nettoyage et enrichissement des données ESS11.

Fonctions appelées dans la partie 1 du notebook `rapport.ipynb` :
téléchargement des fichiers, codes de non-réponse, heatmap des valeurs
manquantes, exclusions, recodage, variables politiques et indices thématiques,
enrichissements (PartyFacts + ParlGov, Eurostat), export de la base propre.
"""

from pathlib import Path
import re
import urllib.error
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Chemins et paramètres
# ---------------------------------------------------------------------------

DOSSIER_DATA = Path("data")
CHEMIN_ESS = DOSSIER_DATA / "ESS11e04_2.csv"
DOSSIER_EXTERNES = DOSSIER_DATA / "externes"
DOSSIER_SORTIE = DOSSIER_DATA                 # la base propre est écrite dans data/
DOSSIER_FIGURES = Path("figures")
CHEMIN_PROPRE = DOSSIER_SORTIE / "ess11_propre.csv"

ANNEE_REFERENCE = 2023
ROUND_ESS = 11           # Utile pour les identifiants PartyFacts

BUCKET = "gleguestensae"
URL_DIFFUSION = f"https://minio.lab.sspcloud.fr/{BUCKET}"

FICHIERS = {
    "ESS11e04_2.csv": CHEMIN_ESS,
    "essprt-all.csv": DOSSIER_EXTERNES / "essprt-all.csv",
    "partyfacts-external-parties.csv": DOSSIER_EXTERNES / "partyfacts-external-parties.csv",
    "view_party.csv": DOSSIER_EXTERNES / "view_party.csv",
}


# ---------------------------------------------------------------------------
# Dictionnaire des variables
# ---------------------------------------------------------------------------

VARIABLES = {
    #Immigration (haut = favorable à l'immigration)
    "imsmetn": dict(nom="immig_meme_ethnie",  vmin=1, vmax=4,  inverser=True,  groupe="immigration"),
    "imdfetn": dict(nom="immig_autre_ethnie", vmin=1, vmax=4,  inverser=True,  groupe="immigration"),
    "impcntr": dict(nom="immig_pays_pauvres", vmin=1, vmax=4,  inverser=True,  groupe="immigration"),
    "imbgeco": dict(nom="immig_economie",     vmin=0, vmax=10, inverser=False, groupe="immigration"),
    "imueclt": dict(nom="immig_culture",      vmin=0, vmax=10, inverser=False, groupe="immigration"),
    "imwbcnt": dict(nom="immig_pays_meilleur",vmin=0, vmax=10, inverser=False, groupe="immigration"),
    #LGBT / mœurs (haut = plus libéral)
    "freehms":  dict(nom="lgbt_libre",          vmin=1, vmax=5, inverser=True,  groupe="lgbt_moeurs"),
    "hmsfmlsh": dict(nom="lgbt_pas_honte",      vmin=1, vmax=5, inverser=False, groupe="lgbt_moeurs"),
    "hmsacld":  dict(nom="lgbt_adoption",       vmin=1, vmax=5, inverser=True,  groupe="lgbt_moeurs"),
    #Redistribution (haut = favorable à la redistribution)
    "gincdif": dict(nom="redistrib_reduire_inegalites", vmin=1, vmax=5, inverser=True, groupe="redistribution"),
    #Écologie (haut = plus préoccupé / engagé)
    "ccnthum": dict(nom="eco_cause_humaine",     vmin=1, vmax=5,  inverser=False, groupe="ecologie"),
    "ccrdprs": dict(nom="eco_responsabilite",    vmin=0, vmax=10, inverser=False, groupe="ecologie"),
    "wrclmch": dict(nom="eco_inquietude",        vmin=1, vmax=5,  inverser=False, groupe="ecologie"),
    #Confiance institutionnelle (haut = confiance)
    "trstprl": dict(nom="confiance_parlement",   vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstlgl": dict(nom="confiance_justice",     vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstplc": dict(nom="confiance_police",      vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstplt": dict(nom="confiance_politiciens", vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstprt": dict(nom="confiance_partis",      vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstep":  dict(nom="confiance_parl_europeen", vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    "trstun":  dict(nom="confiance_onu",         vmin=0, vmax=10, inverser=False, groupe="confiance_institutionnelle"),
    #Europe (haut = pour plus d'intégration)
    "euftf": dict(nom="europe_integration", vmin=0, vmax=10, inverser=False, groupe="europe"),
    #Satisfaction (haut = satisfait)
    "stfeco": dict(nom="satisf_economie",    vmin=0, vmax=10, inverser=False, groupe="satisfaction"),
    "stfgov": dict(nom="satisf_gouvernement",vmin=0, vmax=10, inverser=False, groupe="satisfaction"),
    "stfdem": dict(nom="satisf_democratie",  vmin=0, vmax=10, inverser=False, groupe="satisfaction"),
    #Religiosité (haut = plus religieux)
    "rlgdgr": dict(nom="relig_degre",       vmin=0, vmax=10, inverser=False, groupe="religiosite"),
    "rlgatnd":dict(nom="relig_assiduite",   vmin=1, vmax=7,  inverser=True,  groupe="religiosite"),
    "pray":   dict(nom="relig_priere",      vmin=1, vmax=7,  inverser=True,  groupe="religiosite"),
    #Santé (haut = meilleure santé physique et mentale)
    "health":  dict(nom="sante_auto_evaluee", vmin=1, vmax=5, inverser=True,  groupe="sante"),
    "hlthhmp": dict(nom="sante_pas_genee",    vmin=1, vmax=3, inverser=False, groupe="sante"),
    #Santé mentale (haut = pas de problèmes de santé mentale)
    "fltdpr":  dict(nom="sante_pas_deprime",     vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "flteeff": dict(nom="sante_pas_effort",      vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "slprl":   dict(nom="sante_sommeil_calme",   vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "wrhpp":   dict(nom="sante_heureux",         vmin=1, vmax=4, inverser=False, groupe="sante"),
    "fltlnl":  dict(nom="sante_pas_seul",        vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "enjlf":   dict(nom="sante_profite_vie",     vmin=1, vmax=4, inverser=False, groupe="sante"),
    "fltsd":   dict(nom="sante_pas_triste",      vmin=1, vmax=4, inverser=True,  groupe="sante"),
    "cldgng":  dict(nom="sante_motive",          vmin=1, vmax=4, inverser=True,  groupe="sante"),
    #Genre (haut = plus égalitaire )
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
    "wbrgwrm": dict(nom="genre_rejet_femmes_morales", vmin=1, vmax=5, inverser=False, groupe="genre_hors_indice"),

    #Variables politiques (hors groupes )
    "lrscale": dict(nom="lrscale", vmin=0, vmax=10, inverser=False, groupe="politique"),
    "polintr": dict(nom="pol_interet", vmin=1, vmax=4, inverser=True, groupe="politisation_interet"),
    # actions politiques : 1 = oui, 2 = non. Après inversion et normalisation, oui = 1
    "contplt":  dict(nom="pol_contact_elu",       vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "donprty":  dict(nom="pol_don_parti",         vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "badge":    dict(nom="pol_badge",             vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "sgnptit":  dict(nom="pol_petition",          vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "pbldmna":  dict(nom="pol_manifestation",     vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "bctprd":   dict(nom="pol_boycott",           vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "pstplonl": dict(nom="pol_post_en_ligne",     vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "volunfp":  dict(nom="pol_benevolat",         vmin=1, vmax=2, inverser=True, groupe="politisation_participation"),
    "vote":     dict(nom="pol_a_vote",            vmin=1, vmax=2, inverser=True, groupe="politisation_partisan"),
    "clsprty":  dict(nom="pol_proche_parti",      vmin=1, vmax=2, inverser=True, groupe="politisation_partisan"),
}

GROUPES_THEMATIQUES = [
    "immigration", "lgbt_moeurs", "redistribution", "ecologie",
    "confiance_institutionnelle", "europe", "satisfaction", "religiosite",
    "genre", "sante",
]

SOCIODEMO = {
    "agea":     ("age",15, 110),
    "yrbrn":    ("annee_naissance",1900,2012),
    "gndr":     ("genre_resp",1,2),     # 1 = homme, 2 = femme
    "eisced":   ("education_isced",1,7),
    "eduyrs":   ("annees_etudes",0,40),
    "hinctnta": ("revenu_decile",1,10),
    "domicil":  ("lieu_vie",1,5),     # 1 = grande ville, 5 = isolé
}

POIDS = ["anweight", "pspwght", "pweight", "dweight"]
IDENTIFIANTS = ["idno", "cntry", "essround"]

PAYS_EXCLUS = ["UA", "IL"]
VARIABLES_EXCLUES = ["hinctnta"]      # Variables trop peu renseignées (voir la heatmap)


# ---------------------------------------------------------------------------
# 1. Récupération et chargement
# ---------------------------------------------------------------------------

def recuperer_donnees():
    for nom, local in FICHIERS.items():
        if local.exists():
            print(f"Déjà présent : {local}")
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"{URL_DIFFUSION}/{nom}"
        try:
            urllib.request.urlretrieve(url, local)
            print(f"Téléchargé : {nom} = {local}")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"Échec du téléchargement de {url} : {e}")


#Charge le CSV ESS et met les noms de colonnes en minuscules.
def charger_ess(chemin=CHEMIN_ESS):
    df = pd.read_csv(chemin, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["cntry"] = df["cntry"].astype("string").str.upper().str.strip()
    print(f"{len(df):,} répondants, {df.shape[1]} colonnes, {df['cntry'].nunique()} pays")
    return df


# ---------------------------------------------------------------------------
# 2. Codes de non-réponse
# ---------------------------------------------------------------------------

def colonnes_presentes(df, liste):
    presentes = [c for c in liste if c in df.columns]
    absentes = [c for c in liste if c not in df.columns]
    if absentes:
        warnings.warn(f"Variables absentes du fichier, ignorées : {absentes}")
    return presentes


def codes_manquants_vers_nan(df):
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


# ---------------------------------------------------------------------------
# 3. Heatmap des valeurs manquantes
# ---------------------------------------------------------------------------

def taux_manquants_par_pays(df,variables):
    variables = colonnes_presentes(df, variables)
    taux = df[variables].isna().groupby(df["cntry"]).mean()
    taux["MEAN"] = taux.mean(axis=1)
    return taux.sort_values("MEAN", ascending=False)


def heatmap_manquants(taux, nom_png="fig01_heatmap_manquants.png"):
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * taux.shape[1]), max(6, 0.35 * taux.shape[0])))
    sns.heatmap(taux, cmap="Reds", vmin=0, vmax=1, cbar_kws={"label": "Share of missing values"},
                linewidths=0.3, linecolor="white", ax=ax)
    ax.set_xlabel("ESS variable")
    ax.set_ylabel("Country (sorted by mean missing rate)")
    ax.set_title("ESS11: share of missing values by country and variable")
    plt.tight_layout()
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig.savefig(DOSSIER_FIGURES / nom_png, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. Exclusions
# ---------------------------------------------------------------------------

def exclure_pays(df, pays_exclus=PAYS_EXCLUS):
    n_avant = len(df)
    df = df[~df["cntry"].isin(pays_exclus)].copy()
    print(f"Pays exclus : {pays_exclus} -> {n_avant - len(df):,} répondants retirés, "
          f"{len(df):,} restants")
    return df


def exclure_variables(df, variables_exclues=VARIABLES_EXCLUES):
    """Retire les variables de la base et des dictionnaires utilisés dans la suite."""
    for v in variables_exclues:
        VARIABLES.pop(v, None)
        SOCIODEMO.pop(v, None)
    df = df.drop(columns=[v for v in variables_exclues if v in df.columns])
    print(f"Variables exclues : {variables_exclues}")
    return df


# ---------------------------------------------------------------------------
# 5. Recodage, normalisation, renommage
# ---------------------------------------------------------------------------

#Crée, pour chaque variable de VARIABLES, une colonne au nom lisible, normalisée sur [0, 1] et dans le sens défini plus haut. `lrscale` reste sur 0-10.
def recoder_et_normaliser(df):
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


# ---------------------------------------------------------------------------
# 6. Variables politiques et indices thématiques
# ---------------------------------------------------------------------------

def noms_du_groupe(groupe, df):
    return [p["nom"] for k, p in VARIABLES.items() if p["groupe"] == groupe and p["nom"] in df.columns]


def moyenne_lignes(df, cols, min_part=0.5):
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
            warnings.warn(f"Groupe « {g} » = aucune variable disponible, indice non créé.")
            continue
        df[f"idx_{g}"] = moyenne_lignes(df, cols)
    return df


# ---------------------------------------------------------------------------
# 7. Enrichissement 1 : partis votés (PartyFacts + ParlGov)
# ---------------------------------------------------------------------------

TOUR_RETENU = {"de": "2", "lt": "1"}   # DE : prtvgde2 (2e voix, liste) ; LT : prtvclt1 (1er vote, parti) : Dans ces pays on se limite à un tour de scrutin particulier


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
        print(f"[PartyFacts] Fichier manquant : {chemin} = étape sautée.")
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
    if "partyfacts_id" in essprt.columns:
        pf = essprt[["ess_id", "partyfacts_id"]]
    else:
        pf_ess = liens.loc[liens["dataset_key"] == "essprtv", ["dataset_party_id", "partyfacts_id"]]
        pf = essprt[["ess_id", "first_ess_id"]].astype("string").merge(
            pf_ess, left_on="first_ess_id", right_on="dataset_party_id", how="left"
        )[["ess_id", "partyfacts_id"]]
    pf = pf.dropna().drop_duplicates("ess_id").astype({"ess_id": "string", "partyfacts_id": "Int64"})

    pg_liens = (liens.loc[liens["dataset_key"] == "parlgov", ["partyfacts_id", "dataset_party_id"]]
                .rename(columns={"dataset_party_id": "parlgov_id"})
                .drop_duplicates("partyfacts_id"))
    pg_liens["parlgov_id"] = pd.to_numeric(pg_liens["parlgov_id"], errors="coerce").astype("Int64")
    pg_liens["partyfacts_id"] = pg_liens["partyfacts_id"].astype("Int64")

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


# ---------------------------------------------------------------------------
# 8. Enrichissement 2 : données pays via l'API Eurostat
# ---------------------------------------------------------------------------

INDICATEURS_EUROSTAT = {
    "pib_hab_spa":   dict(code="tec00114", filtres={}),
    "gini": dict(code="ilc_di12", filtres={"age": "TOTAL"}),
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
        print("[Eurostat] Package `eurostat` non installé (pip install eurostat) = étape sautée.")
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


# ---------------------------------------------------------------------------
# 9. Dictionnaire des variables et export
# ---------------------------------------------------------------------------

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


def exporter(df, chemin=CHEMIN_PROPRE):
    df.to_csv(chemin, index=False)
    dictionnaire_variables(df).to_csv(DOSSIER_SORTIE / "dictionnaire_variables.csv", index=False)
    print(f"Export : {chemin}")


# ---------------------------------------------------------------------------
# 10. Statistiques descriptives de la base finale
# ---------------------------------------------------------------------------

NOMS_PAYS = {
    "AT": "Autriche", "BE": "Belgique", "BG": "Bulgarie", "CH": "Suisse", "CY": "Chypre",
    "DE": "Allemagne", "EE": "Estonie", "ES": "Espagne", "FI": "Finlande", "FR": "France",
    "GB": "Royaume-Uni", "GR": "Grèce", "HR": "Croatie", "HU": "Hongrie", "IE": "Irlande",
    "IS": "Islande", "IT": "Italie", "LT": "Lituanie", "LV": "Lettonie", "ME": "Monténégro",
    "NL": "Pays-Bas", "NO": "Norvège", "PL": "Pologne", "PT": "Portugal", "RS": "Serbie",
    "SE": "Suède", "SI": "Slovénie", "SK": "Slovaquie",
}

VARIABLES_POLITIQUES = {
    "lrscale": "Position gauche-droite (%)",
    "politiscale": "Politisation (%)",
    "parti_lr": "Parti voté relié à ParlGov (%)",
}


def stats_descriptives_pays(propre):
    """Effectif et part de réponses valides par pays pour les principales variables."""
    variables = {v: lib for v, lib in VARIABLES_POLITIQUES.items() if v in propre.columns}
    # Part de réponses valides par pays
    taux = (propre[list(variables)].notna()
            .groupby(propre["cntry"], observed=True).mean().mul(100).round(1)
            .rename(columns=variables))

    stats_pays = pd.concat(
        [propre.groupby("cntry", observed=True).size().rename("Répondants"), taux], axis=1
    )

    # Noms de pays en clair, tri par effectif décroissant
    stats_pays.index = stats_pays.index.map(lambda c: NOMS_PAYS.get(c, c))
    stats_pays.index.name = "Pays"
    stats_pays = stats_pays.sort_values("Répondants", ascending=False)

    # Ligne : moyenne de chaque colonne sur l'ensemble des pays (ajoutée après le tri pour rester en bas)
    moyenne_pays = stats_pays.mean().round(1)
    moyenne_pays["Répondants"] = round(moyenne_pays["Répondants"])
    stats_pays.loc["Moyenne des pays"] = moyenne_pays

    stats_pays.to_csv(DOSSIER_SORTIE / "stats_descriptives_par_pays.csv")
    print(f"{len(stats_pays) - 1} pays, {len(propre):,} répondants")
    return stats_pays
