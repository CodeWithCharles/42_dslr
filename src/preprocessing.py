"""
Module partage train/predict
"""

from __future__ import annotations

import loader
import math_utils
import pandas as pd

EXCLUDED: set[str] = {"Index"}

def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV (délègue à loader.load_csv)."""
    return loader.load_csv(path)


def select_features(df, features: list[str] | None = None) -> tuple[list[list[float]], list[str]]:
    """Sélectionne les colonnes-features et les renvoie orientées COLONNES.

    - features None (TRAIN) : dérive les colonnes numériques via
      loader.numeric_columns(df), moins EXCLUDED.
    - features fourni (PREDICT) : utilise cette liste telle quelle, pour
      garantir le même ensemble/ordre que le train.

    Retourne (colonnes, noms_résolus) : une liste par feature (NaN possibles
    à ce stade) et la liste des noms effectivement retenus, à sauvegarder
    avec les poids.
    """
    if features is None:
        features = [c for c in loader.numeric_columns(df) if c not in EXCLUDED]
    columns = [df[name].tolist() for name in features]
    return columns, features

def fit_params(columns: list[list[float]]) -> tuple[list[float], list[float]]:
    """TRAIN uniquement : calcule (means, stds) par colonne sur les valeurs
    brutes (NaN ignorés par math_utils). Std AVANT tout remplissage de NaN
    pour ne pas fausser la variance."""
    means = [math_utils.mean(col) for col in columns]
    stds = [math_utils.std(col) for col in columns]
    return means, stds

def transform(columns: list[list[float]],
              means: list[float],
              stds: list[float]) -> list[list[float]]:
    """Applique la normalisation avec des params FOURNIS puis transpose.

    Par colonne : remplace les NaN par means[i], puis standardise avec
    (means[i], stds[i]). Enfin transpose : une ligne = un étudiant.
    """
    standardized_cols: list[list[float]] = []
    for i, col in enumerate(columns):
        filled = math_utils.fill_na_with_mean(col, means[i])
        std_col, _, _ = math_utils.standardize(filled, means[i], stds[i])
        standardized_cols.append(std_col)
    rows = [list(row) for row in zip(*standardized_cols)]
    return rows