"""
Fonctions communes aux trois parties du projet.

La corrélation pondérée est utilisée à la fois dans la partie 2 (corrélations
pays par pays) et dans la partie 3 (corrélations par tranche d'âge) : on n'en
garde qu'une seule version, ici.
"""

import numpy as np


POIDS = "anweight"      # poids d'analyse recommandé par l'ESS
N_MIN = 100             # nombre minimal de réponses pour calculer une corrélation


def moyenne_ponderee(x, w):
    """Moyenne de x pondérée par w, en ignorant les valeurs manquantes."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    return np.average(x[ok], weights=w[ok])


def corr_ponderee(x, y, w, n_min=N_MIN):
    """Corrélation de Pearson entre x et y, pondérée par w.

    On garde seulement les personnes qui ont répondu aux deux questions (et
    dont le poids est positif). Si elles sont moins de `n_min`, on renvoie NaN.
    """
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < n_min:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    ecart_x = x - np.average(x, weights=w)
    ecart_y = y - np.average(y, weights=w)
    covariance = np.average(ecart_x * ecart_y, weights=w)
    variance_x = np.average(ecart_x ** 2, weights=w)
    variance_y = np.average(ecart_y ** 2, weights=w)
    if variance_x == 0 or variance_y == 0:
        return np.nan
    return covariance / np.sqrt(variance_x * variance_y)
