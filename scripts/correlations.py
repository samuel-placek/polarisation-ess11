"""
Partie 2 - Polarisation politique et attitudes (ESS11).

Fonctions utilisées par le notebook `partie2.ipynb`. On part de la base
nettoyée par la partie 1 (`output/ess11_propre.csv`).

Idée générale : on calcule des corrélations pondérées entre variables,
pays par pays, puis on fait la moyenne sur les pays (chaque pays compte autant).
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

CHEMIN_BASE = Path("output") / "ess11_propre.csv"
DOSSIER_FIGURES = Path("figures")

PAYS_EXCLUS = ["IL"]          # Israël : on reste sur un périmètre européen
POIDS = "anweight"            # poids d'analyse recommandé par l'ESS
N_MIN = 100                   # nombre minimal de réponses pour calculer une corrélation
CONTROLES = ["age", "femme", "education_isced", "urbain"]   # pour les corrélations partielles

BLEU, ORANGE, GRIS = "#2a78d6", "#eb6834", "#52514e"

# Préfixe des noms de variables (créés en partie 1) -> nom du groupe thématique
GROUPES = {
    "immig_": "Immigration",
    "redistrib_": "Redistribution",
    "lgbt_": "LGBT rights",
    "eco_": "Climate change",
    "genre_": "Gender equality",
    "europe_": "European integration",
    "relig_": "Religiosity",
    "satisf_": "Satisfaction",
    "confiance_": "Institutional trust",
    "sante_": "Health & well-being",
}

# Libellés anglais des items (une valeur haute = le libellé est vrai).
# Comme en partie 1, on n'utilise pas `genre_rejet_femmes_morales`, qui n'est
# pas corrélé aux autres questions sur le genre.
LIBELLES = {
    "immig_meme_ethnie": "Allow immigrants of same ethnicity",
    "immig_autre_ethnie": "Allow immigrants of different ethnicity",
    "immig_pays_pauvres": "Allow immigrants from poorer countries",
    "immig_economie": "Immigration good for the economy",
    "immig_culture": "Immigrants enrich cultural life",
    "immig_pays_meilleur": "Immigrants make country a better place",
    "redistrib_reduire_inegalites": "Government should reduce income gaps",
    "lgbt_libre": "Gays & lesbians free to live as they wish",
    "lgbt_pas_honte": "Not ashamed if relative is gay",
    "lgbt_adoption": "Same-sex couples may adopt",
    "eco_cause_humaine": "Climate change caused by humans",
    "eco_responsabilite": "Feels responsible for climate change",
    "eco_inquietude": "Worried about climate change",
    "genre_emploi_egal_famille": "Equal employment good for family life",
    "genre_direction_egale": "Equal management good for business",
    "genre_salaire_egal_eco": "Equal pay good for the economy",
    "genre_parite_parlement": "Gender parity in parliament",
    "genre_conge_parental_egal": "Equal parental leave",
    "genre_licencier_insultes": "Fire employees insulting women",
    "genre_amende_inegalite": "Fine firms paying women less",
    "genre_rejet_femmes_pouvoir": "Rejects 'women seek power over men'",
    "genre_rejet_femmes_offensees": "Rejects 'women easily offended'",
    "genre_rejet_harcelement_exagere": "Rejects 'women exaggerate harassment'",
    "genre_percoit_ecart_salaire": "Perceives a gender pay gap",
    "genre_rejet_protection": "Rejects 'women should be protected by men'",
    "europe_integration": "EU unification should go further",
    "relig_degre": "Religious (self-rated)",
    "relig_assiduite": "Attends religious services",
    "relig_priere": "Prays",
    "satisf_economie": "Satisfied with the economy",
    "satisf_gouvernement": "Satisfied with the government",
    "satisf_democratie": "Satisfied with democracy",
    "confiance_parlement": "Trust in parliament",
    "confiance_justice": "Trust in the legal system",
    "confiance_police": "Trust in the police",
    "confiance_politiciens": "Trust in politicians",
    "confiance_partis": "Trust in political parties",
    "confiance_parl_europeen": "Trust in the European Parliament",
    "confiance_onu": "Trust in the United Nations",
    "sante_auto_evaluee": "Good self-rated health",
    "sante_pas_genee": "Not hampered by illness",
    "sante_pas_deprime": "Not depressed",
    "sante_pas_effort": "Not 'everything an effort'",
    "sante_sommeil_calme": "Restful sleep",
    "sante_heureux": "Felt happy",
    "sante_pas_seul": "Not lonely",
    "sante_profite_vie": "Enjoyed life",
    "sante_pas_triste": "Not sad",
    "sante_motive": "Could get going",
}


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def charger_base(chemin=CHEMIN_BASE):
    """Charge la base de la partie 1 et retire les pays exclus."""
    df = pd.read_csv(chemin, low_memory=False)
    df = df[~df["cntry"].isin(PAYS_EXCLUS)].reset_index(drop=True)
    print(f"{len(df):,} répondants, {df['cntry'].nunique()} pays")
    return df


def groupe_de(item):
    """Groupe thématique d'un item, trouvé grâce au début de son nom."""
    for prefixe, groupe in GROUPES.items():
        if item.startswith(prefixe):
            return groupe


def liste_items(df):
    """Les 49 items thématiques, rangés dans l'ordre des groupes."""
    ordre = list(GROUPES.values())
    items = [c for c in LIBELLES if c in df.columns]
    return sorted(items, key=lambda c: ordre.index(groupe_de(c)))


# ---------------------------------------------------------------------------
# 2. Statistiques descriptives
# ---------------------------------------------------------------------------

def moyenne_ponderee(x, w):
    """Moyenne de x pondérée par w, en ignorant les valeurs manquantes."""
    ok = x.notna()
    return np.average(x[ok], weights=w[ok])


def tableau_pays(df):
    """Pour chaque pays : effectif, non-réponse à l'échelle gauche-droite,
    position moyenne et extrémité moyenne (pondérées)."""
    lignes = []
    for pays, g in df.groupby("cntry"):
        lignes.append({
            "country": pays,
            "N": len(g),
            "no_answer_lr_%": 100 * g["lrscale"].isna().mean(),
            "mean_position": moyenne_ponderee(g["lrscale"], g[POIDS]),
            "mean_extremity": moyenne_ponderee(g["lrscaleabs"], g[POIDS]),
        })
    tableau = pd.DataFrame(lignes).set_index("country")
    return tableau.sort_values("mean_extremity", ascending=False).round(2)


# ---------------------------------------------------------------------------
# 3. Corrélations
# ---------------------------------------------------------------------------

def corr_ponderee(x, y, w):
    """Corrélation de Pearson entre x et y, pondérée par w.
    On garde seulement les personnes qui ont répondu aux deux questions."""
    ok = x.notna() & y.notna()
    if ok.sum() < N_MIN:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    ecart_x = x - np.average(x, weights=w)
    ecart_y = y - np.average(y, weights=w)
    covariance = np.average(ecart_x * ecart_y, weights=w)
    variance_x = np.average(ecart_x ** 2, weights=w)
    variance_y = np.average(ecart_y ** 2, weights=w)
    return covariance / np.sqrt(variance_x * variance_y)


def corr_par_pays(df, a, b):
    """Corrélation entre les variables a et b dans chaque pays (une valeur par pays)."""
    return df.groupby("cntry")[[a, b, POIDS]].apply(lambda g: corr_ponderee(g[a], g[b], g[POIDS]))


def residus_controles(df, variables):
    """Pour chaque variable et dans chaque pays, on retire la partie expliquée
    par les contrôles (âge, sexe, éducation, urbain) avec une régression linéaire.
    Corréler ensuite deux résidus donne la corrélation partielle."""
    residus = df[["cntry", POIDS]].copy()
    for v in variables:
        residus[v] = np.nan
        for pays, g in df.groupby("cntry"):
            g = g.dropna(subset=[v] + CONTROLES)
            X = np.column_stack([np.ones(len(g))] + [g[c] for c in CONTROLES])
            coef = np.linalg.lstsq(X, g[v], rcond=None)[0]
            residus.loc[g.index, v] = g[v] - X @ coef
    return residus


def table_paires(df, df_residus, items):
    """Pour chaque paire d'items : corrélation brute et partielle,
    en moyenne sur les pays."""
    lignes = []
    for a, b in combinations(items, 2):
        lignes.append({
            "item_1": LIBELLES[a], "group_1": groupe_de(a),
            "item_2": LIBELLES[b], "group_2": groupe_de(b),
            "r": corr_par_pays(df, a, b).mean(),
            "r_partial": corr_par_pays(df_residus, a, b).mean(),
        })
    paires = pd.DataFrame(lignes)
    paires["same_group"] = paires["group_1"] == paires["group_2"]
    return paires


def table_position_pays(df, items):
    """Tableau items x pays : corrélation de chaque item avec lrscale."""
    return pd.DataFrame({LIBELLES[i]: corr_par_pays(df, i, "lrscale") for i in items}).T


def intensite(df, items):
    """Intensité de l'opinion : distance à la moyenne du pays, en écarts-types,
    sans tenir compte du sens (0 = opinion moyenne, grand = opinion tranchée)."""
    resultat = df[["cntry", POIDS, "lrscaleabs"]].copy()
    for v in items:
        moyenne = df.groupby("cntry")[v].transform("mean")
        ecart_type = df.groupby("cntry")[v].transform("std")
        resultat[v] = ((df[v] - moyenne) / ecart_type).abs()
    return resultat


def table_bg(table_pays_lr, df_intensite, items):
    """Données de la figure de Baldassarri & Gelman, une ligne par item :
    - r_position : corrélation moyenne avec la position gauche-droite. Comme chez
      B&G, on retourne l'item si besoin pour que cette corrélation soit positive
      (« droite » = valeur haute) ; les items retournés sont marqués « (rev.) » ;
    - r_extremity : corrélation moyenne entre l'extrémité et l'intensité de l'opinion."""
    lignes = []
    for i in items:
        r_position = table_pays_lr.loc[LIBELLES[i]].mean()
        label = LIBELLES[i]
        if r_position < 0:
            r_position, label = -r_position, label + "  (rev.)"
        lignes.append({
            "label": label,
            "group": groupe_de(i),
            "r_position": r_position,
            "r_extremity": corr_par_pays(df_intensite, i, "lrscaleabs").mean(),
        })
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------

def enregistrer(fig, nom):
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig.savefig(DOSSIER_FIGURES / nom, dpi=200, bbox_inches="tight")


def figure_polarisation(df, tableau):
    """(A) Répartition des positions gauche-droite ; (B) extrémité moyenne par pays."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    parts = 100 * df.groupby("lrscale")[POIDS].sum() / df.dropna(subset=["lrscale"])[POIDS].sum()
    ax1.bar(parts.index, parts.values, color=BLEU)
    ax1.set_xticks(range(11))
    ax1.set_xlabel("Left-right self-placement (0 = left, 10 = right)")
    ax1.set_ylabel("Share of respondents (%)")
    ax1.set_title("(A) Distribution of left-right position", loc="left")

    t = tableau.sort_values("mean_extremity")
    ax2.barh(t.index, t["mean_extremity"], color=ORANGE)
    ax2.axvline(t["mean_extremity"].mean(), color=GRIS, linestyle="--", label="average")
    ax2.set_xlabel("Mean ideological extremity |lrscale - 5| (0 to 5)")
    ax2.set_title("(B) Mean extremity by country", loc="left")
    ax2.legend(frameon=False, loc="lower right")

    sns.despine()
    fig.tight_layout()
    enregistrer(fig, "fig1_polarisation.png")
    return parts


def figure_dellaposta(paires):
    """Figure 1 de DellaPosta et al. : densité des |r| entre toutes les paires d'items."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    sns.kdeplot(paires["r"].abs(), ax=ax1, color=BLEU, clip=(0, 1), label="Zero-order")
    sns.kdeplot(paires["r_partial"].abs(), ax=ax1, color=ORANGE, linestyle="--", clip=(0, 1), label="Partial")
    ax1.set_title(f"(A) All {len(paires)} item pairs", loc="left")

    meme = paires[paires["same_group"]]
    autre = paires[~paires["same_group"]]
    sns.kdeplot(meme["r"].abs(), ax=ax2, color=BLEU, clip=(0, 1), label="Same domain")
    sns.kdeplot(autre["r"].abs(), ax=ax2, color=ORANGE, linestyle="--", clip=(0, 1), label="Different domains")
    ax2.set_title("(B) Zero-order, by type of pair", loc="left")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("Pairwise correlation magnitude |r|")
        ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    enregistrer(fig, "fig2_dellaposta.png")


def figure_heatmap(table_pays_lr, items):
    """Heatmap : corrélation de chaque item avec lrscale, par pays, plus la moyenne."""
    ordre_pays = table_pays_lr.abs().mean().sort_values(ascending=False).index
    data = table_pays_lr[ordre_pays]
    data.insert(0, "EUROPE", table_pays_lr.mean(axis=1))

    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(data, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5, ax=ax,
                cbar_kws={"label": "Correlation with left-right position (red = more common on the right)",
                          "orientation": "horizontal", "shrink": 0.5, "pad": 0.08})
    ax.axvline(1, color="white", linewidth=4)          # sépare la colonne EUROPE

    # lignes blanches entre les groupes + nom du groupe à droite
    groupes = [groupe_de(i) for i in items]
    for k in range(len(items)):
        if k == 0 or groupes[k] != groupes[k - 1]:
            ax.axhline(k, color="white", linewidth=3)
            ax.text(len(data.columns) + 0.3, k + 0.5, groupes[k], va="center", fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Correlation between each item and left-right position, by country\n"
                 "(countries sorted by mean |r|, strongest on the left)", loc="left")
    fig.tight_layout()
    enregistrer(fig, "fig3_heatmap_pays.png")


def figure_bg(table):
    """Figure 2 de Baldassarri & Gelman : un point par item, regroupés par domaine."""
    ordre_groupes = table.groupby("group")["r_position"].mean().sort_values(ascending=False).index
    table = table.copy()
    table["rang_groupe"] = table["group"].map({g: k for k, g in enumerate(ordre_groupes)})
    table = table.sort_values(["rang_groupe", "r_position"], ascending=[True, False]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 13))
    y = np.arange(len(table))
    ax.scatter(table["r_position"], y, color=BLEU, label="Item x left-right position", zorder=3)
    ax.scatter(table["r_extremity"], y, color=ORANGE, label="Opinion intensity x extremity", zorder=3)
    ax.set_yticks(y, table["label"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=GRIS, linewidth=0.8)
    ax.grid(axis="y", color="#e4e3df")
    x_max = table[["r_position", "r_extremity"]].max().max() + 0.09
    ax.set_xlim(-0.01, x_max)          # place libre à droite pour le nom des domaines

    # ligne de séparation + nom du domaine au début de chaque groupe
    for k in range(len(table)):
        if k == 0 or table.loc[k, "group"] != table.loc[k - 1, "group"]:
            ax.axhline(k - 0.5, color=GRIS, linewidth=0.8)
            ax.text(x_max, k - 0.3, table.loc[k, "group"].upper(), ha="right", va="top",
                    fontsize=8, fontweight="bold", color=GRIS)

    ax.set_xlabel("Average within-country correlation")
    ax.set_title("Correlations of issue attitudes with position and extremity", loc="left")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(left=True)
    fig.tight_layout()
    enregistrer(fig, "fig4_baldassarri_gelman.png")
