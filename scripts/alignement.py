"""
Partie 3 - Mécanisme théorique de l'alignement et dynamique par âge et par thème.

Fonctions appelées dans la partie 3 du notebook `rapport.ipynb` :
simulation de la figure 1 de Baldassarri & Gelman (2008), alignement par
groupe thématique selon l'âge (adaptation de Kozlowski & Murphy, 2021) et
tests de robustesse.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import BSpline
from scipy import stats

from .outils import corr_ponderee


DOSSIER_FIGURES = Path("figures")

PARAMS = {
    "seed": 42,
    "bg_n_samples": 25000,
    "bg_n_dist": 800,
    "bg_rhos_density": [0.0, 0.25, 0.5, 0.75, 1.0],
    "ess_min_n": 30,
    "age_bins": [14, 24, 34, 44, 54, 64, 74, 120],
    "age_labels": ["15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"],
    "spline_df": 3,
    "dpi": 300,
}

# Reprise exacte des 10 groupes thématiques de la Partie 1
GROUPES_ITEMS = {
    "immigration": ["immig_meme_ethnie", "immig_autre_ethnie", "immig_pays_pauvres",
                    "immig_economie", "immig_culture", "immig_pays_meilleur"],
    "lgbt_moeurs": ["lgbt_libre", "lgbt_pas_honte", "lgbt_adoption"],
    "redistribution": ["redistrib_reduire_inegalites"],
    "ecologie": ["eco_cause_humaine", "eco_responsabilite", "eco_inquietude"],
    "europe": ["europe_integration"],
    "confiance_institutionnelle": ["confiance_parlement", "confiance_justice", "confiance_police",
                                   "confiance_politiciens", "confiance_partis", "confiance_parl_europeen", "confiance_onu"],
    "satisfaction": ["satisf_economie", "satisf_gouvernement", "satisf_democratie"],
    "religiosite": ["relig_degre", "relig_assiduite", "relig_priere"],
    "genre": ["genre_emploi_egal_famille", "genre_direction_egale", "genre_salaire_egal_eco",
              "genre_parite_parlement", "genre_conge_parental_egal", "genre_licencier_insultes",
              "genre_amende_inegalite", "genre_rejet_femmes_pouvoir", "genre_rejet_femmes_offensees",
              "genre_rejet_harcelement_exagere", "genre_percoit_ecart_salaire", "genre_rejet_protection"],
    "sante": ["sante_auto_evaluee", "sante_pas_genee", "sante_pas_deprime", "sante_pas_effort",
              "sante_sommeil_calme", "sante_heureux", "sante_pas_seul", "sante_profite_vie",
              "sante_pas_triste", "sante_motive"],
}

LABELS_GROUPES = {
    "immigration": "Immigration",
    "lgbt_moeurs": "LGBT / Mœurs",
    "redistribution": "Redistribution",
    "ecologie": "Écologie",
    "europe": "Intégration européenne",
    "confiance_institutionnelle": "Confiance institutionnelle",
    "satisfaction": "Satisfaction politique",
    "religiosite": "Religiosité",
    "genre": "Égalité de genre",
    "sante": "Santé physique & mentale",
}

# Style des figures de la partie 3 (appliqué seulement à ces figures)
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def enregistrer(fig, nom):
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig.savefig(DOSSIER_FIGURES / nom, dpi=PARAMS["dpi"], bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. Chargement et sous-échantillon analytique
# ---------------------------------------------------------------------------

def charger_base_partie3(chemin_csv):
    """Garde les répondants avec un âge et une position gauche-droite valides,
    et crée les tranches d'âge décennales. Renvoie la base, le poids utilisé
    et le tableau des effectifs par tranche d'âge."""
    df_init = pd.read_csv(chemin_csv, low_memory=False)
    weight_col = next((w for w in ["anweight", "pspwght", "dweight"] if w in df_init.columns), None)
    if weight_col is None:
        df_init["unit_weight"] = 1.0
        weight_col = "unit_weight"

    df = df_init[df_init["age"].between(15, 105) & df_init["lrscale"].between(0, 10)].copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    df["age_bin_p3"] = pd.cut(df["age"], bins=PARAMS["age_bins"], labels=PARAMS["age_labels"])

    print(f"Base Partie 1 : {len(df_init):,} répondants -> Sous-échantillon Partie 3 (âge & lrscale valides) : "
          f"{len(df):,} ({len(df)/len(df_init):.1%}) | Pondération : {weight_col}")
    effectifs = df.groupby("age_bin_p3", observed=False)["age"].agg(Effectif="count", Age_Moyen="mean").round(1)
    return df, weight_col, effectifs


# ---------------------------------------------------------------------------
# 2. Simulation de Baldassarri & Gelman (2008), figure 1
# ---------------------------------------------------------------------------

def generate_bg_figure1(nom_png="fig06_bg_simulation.png"):
    rng = np.random.default_rng(PARAMS["seed"])
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panneau A : Densité du score idéologique moyen
        grid_x = np.linspace(-3, 3, 400)
        for rho, ls in zip(PARAMS["bg_rhos_density"], linestyles):
            cov = np.full((4, 4), rho)
            np.fill_diagonal(cov, 1.0)
            mean_score = rng.multivariate_normal(np.zeros(4), cov, size=PARAMS["bg_n_samples"]).mean(axis=1)
            bw = max(1.06 * np.std(mean_score, ddof=1) * (len(mean_score) ** (-0.2)), 0.03)
            kde = np.exp(-0.5 * ((grid_x[:, None] - mean_score[None, :]) / bw) ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
            axes[0].plot(grid_x, kde, linestyle=ls, color="black", linewidth=1.6, label=rf"$\rho = {rho}$")

        axes[0].set_title("(A) Distribution du score moyen (4 dimensions)")
        axes[0].set_xlabel("Position moyenne sur les 4 dimensions")
        axes[0].set_ylabel("Densité")
        axes[0].set_xlim(-3, 3)
        axes[0].legend(title="Corrélation", frameon=False)

        # Panneau B : Distance de Manhattan moyenne
        rho_grid = np.linspace(0.0, 1.0, 11)
        all_d, bet_d, wit_d = [], [], []
        for rho in rho_grid:
            cov = np.full((4, 4), rho)
            np.fill_diagonal(cov, 1.0)
            X = rng.multivariate_normal(np.zeros(4), cov, size=PARAMS["bg_n_dist"])
            clusters = (X.mean(axis=1) >= 0).astype(int)
            dmat = squareform(pdist(X, metric="cityblock"))
            iu = np.triu_indices(len(X), k=1)
            vals, same = dmat[iu], (clusters[iu[0]] == clusters[iu[1]])
            all_d.append(vals.mean())
            bet_d.append(vals[~same].mean())
            wit_d.append(vals[same].mean())

        axes[1].plot(rho_grid, bet_d, "-", color="black", lw=1.8, label="Entre clusters")
        axes[1].plot(rho_grid, all_d, "--", color="black", lw=1.4, label="Population totale")
        axes[1].plot(rho_grid, wit_d, ":", color="black", lw=1.8, label="Au sein des clusters")
        axes[1].set_title("(B) Distance de Manhattan moyenne")
        axes[1].set_xlabel(r"Corrélation entre dimensions ($\rho$)")
        axes[1].set_ylabel("Distance moyenne")
        axes[1].legend(frameon=False)

        plt.tight_layout()
        enregistrer(fig, nom_png)
    return fig


# ---------------------------------------------------------------------------
# 3. Alignement par groupe thématique selon l'âge (Kozlowski & Murphy, 2021)
# ---------------------------------------------------------------------------

def fit_predict_bspline_by_group(corr_df, age_grid, df_spline=3, alpha=0.05):
    groups = sorted(corr_df["groupe"].unique())
    x_all, y_all = corr_df["age_mid"].to_numpy(), corr_df["abs_corr"].to_numpy()
    degree = 3
    n_int = max(0, df_spline - degree)
    int_knots = np.quantile(x_all, np.linspace(0, 1, n_int + 2)[1:-1]) if n_int > 0 else np.array([])
    knots = np.concatenate(([x_all.min()] * (degree + 1), int_knots, [x_all.max()] * (degree + 1)))

    B_obs = BSpline.design_matrix(x_all, knots, degree).toarray()[:, 1:]
    B_grid = BSpline.design_matrix(age_grid, knots, degree).toarray()[:, 1:]

    X_design = np.hstack([
        np.hstack([ind := (corr_df["groupe"].to_numpy() == g).astype(float)[:, None], ind * B_obs])
        for g in groups
    ])
    beta, _, _, _ = np.linalg.lstsq(X_design, y_all, rcond=None)
    resid = y_all - X_design @ beta
    dof = max(len(y_all) - np.linalg.matrix_rank(X_design), 1)
    cov_beta = (np.sum(resid**2) / dof) * np.linalg.pinv(X_design.T @ X_design)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=dof)

    preds = {}
    for idx_g, g in enumerate(groups):
        X_grid = np.hstack([
            np.hstack([np.ones((len(age_grid), 1)), B_grid]) if j == idx_g
            else np.zeros((len(age_grid), 1 + B_grid.shape[1]))
            for j in range(len(groups))
        ])
        mean_pred = X_grid @ beta
        se_pred = np.sqrt(np.maximum(np.sum((X_grid @ cov_beta) * X_grid, axis=1), 0.0))
        preds[g] = {"mean": mean_pred, "ci_lower": mean_pred - t_crit * se_pred, "ci_upper": mean_pred + t_crit * se_pred}
    return preds


def analyser_alignement_age_thematiques(df, groupes_dict, weight_col, target_col="lrscale",
                                        titre="", ylabel="", nom_png=None):
    """Corrélation pondérée |r| de chaque item avec `target_col`, par tranche d'âge,
    puis lissage par groupe thématique. Renvoie les corrélations et le tableau résumé."""
    records = [
        {"age_mid": grp["age"].mean(), "groupe": g, "label": LABELS_GROUPES.get(g, g),
         "abs_corr": abs(r)}
        for _, grp in df.groupby("age_bin_p3", observed=False)
        for g, cols in groupes_dict.items()
        for col in cols if col in grp.columns
        if np.isfinite(r := corr_ponderee(
            pd.to_numeric(grp[col], errors="coerce"),
            pd.to_numeric(grp[target_col], errors="coerce"),
            grp[weight_col],
            n_min=PARAMS["ess_min_n"],
        ))
    ]
    corr_df = pd.DataFrame(records)
    summary = (corr_df.groupby("label")["abs_corr"]
               .agg(Moyenne="mean", Ecart_Type="std", Mediane="median", Max="max")
               .sort_values("Moyenne", ascending=False))

    age_grid = np.linspace(corr_df["age_mid"].min(), corr_df["age_mid"].max(), 120)
    preds = fit_predict_bspline_by_group(corr_df, age_grid, df_spline=PARAMS["spline_df"])

    panneaux = [
        (["immigration", "lgbt_moeurs", "redistribution", "ecologie", "europe"], "(A) Enjeux politiques, culturels et sociétaux"),
        (["confiance_institutionnelle", "satisfaction", "religiosite", "genre", "sante"], "(B) Institutions, valeurs, genre et santé"),
    ]
    palette = ["#1b4f72", "#922b21", "#1e8449", "#7d6608", "#5b2c6f"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

        for ax, (liste_g, sous_titre) in zip(axes, panneaux):
            for idx, g in enumerate(liste_g):
                if g not in preds:
                    continue
                c, res = palette[idx % len(palette)], preds[g]
                ax.plot(age_grid, res["mean"], color=c, lw=2.1, label=LABELS_GROUPES.get(g, g))
                ax.fill_between(age_grid, res["ci_lower"], res["ci_upper"], color=c, alpha=0.12)
                pts = corr_df[corr_df["groupe"] == g].groupby("age_mid")["abs_corr"].mean().reset_index()
                ax.scatter(pts["age_mid"], pts["abs_corr"], color=c, s=20, alpha=0.8)
            ax.set_title(sous_titre)
            ax.set_xlabel("Âge moyen de la tranche (années)")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, loc="upper left")

        axes[0].set_ylabel(ylabel)
        fig.suptitle(titre, fontsize=11.5, y=1.02)
        plt.tight_layout()
        if nom_png:
            enregistrer(fig, nom_png)
    return corr_df, summary.round(3)


def effectifs_partie3(df_ess):
    """Nombre de pays et effectifs utilisés dans la partie 3."""
    return pd.Series({
        "Nombre de pays": df_ess["cntry"].nunique(),
        "N (lrscale valide)": int(df_ess["lrscale"].notna().sum()),
        "N ParlGov (parti voté relié)": int(df_ess["parti_lr"].notna().sum()) if "parti_lr" in df_ess else 0,
    })
