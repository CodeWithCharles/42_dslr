"""
preprocessing.py — Preparation des features partagee entre l'entrainement
(logreg_train.py) et la prediction (logreg_predict.py).

Produit la matrice de features X de facon identique des deux cotes, avec les
memes moyennes/ecarts-types (calcules sur le train, reutilises au test).
Separe fit (apprentissage des parametres, train uniquement) et transform
(application, train et test). Aucun calcul statistique brut ici : tout est
delegue a math_utils. Voir doc/common.md pour le socle partage et
doc/preprocessing.md pour le fonctionnement propre a ce module.
"""

from __future__ import annotations

import pandas as pd

from loader import load_csv, numeric_columns
from math_utils import fill_na_with_mean, mean, standardize, std

EXCLUDED: set[str] = {"Index"}


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV (delegue a loader.load_csv)."""
    return load_csv(path)


def select_features(
    df, features: list[str] | None = None, excluded: list[str] | None = None
) -> tuple[list[list[float]], list[str]]:
    """Selectionne les colonnes-features et les renvoie orientees COLONNES.

    - features None (TRAIN) : derive les colonnes numeriques via
      numeric_columns(df), moins EXCLUDED.
    - features fourni (PREDICT) : utilise cette liste telle quelle, pour
      garantir le meme ensemble/ordre que le train.

    Retourne (colonnes, noms_resolus) : une liste par feature (NaN possibles
    a ce stade) et la liste des noms effectivement retenus, a sauvegarder
    avec les poids.
    """
    if features is None:
        features = [c for c in numeric_columns(df) if c not in (EXCLUDED if excluded is None else excluded)]
    columns = [df[name].tolist() for name in features]
    return columns, features


def fit_params(
    columns: list[list[float]],
) -> tuple[list[float], list[float]]:
    """TRAIN uniquement : calcule (means, stds) par colonne sur les valeurs
    brutes (NaN ignores par math_utils). Std AVANT tout remplissage de NaN
    pour ne pas fausser la variance."""
    means = [mean(col) for col in columns]
    stds = [std(col) for col in columns]
    return means, stds


def transform(
    columns: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    """Applique la normalisation avec des params FOURNIS puis transpose.

    Par colonne : remplace les NaN par means[i], puis standardise avec
    (means[i], stds[i]). Enfin transpose : une ligne = un etudiant.
    """
    standardized_cols: list[list[float]] = []
    for i, col in enumerate(columns):
        filled = fill_na_with_mean(col, means[i])
        std_col, _, _ = standardize(filled, means[i], stds[i])
        standardized_cols.append(std_col)
    return [list(row) for row in zip(*standardized_cols)]