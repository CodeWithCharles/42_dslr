"""
math_utils.py — Fonctions mathematiques et de manipulation d'array du
projet DSLR.

Aucune dependance externe (uniquement la stdlib) : ce module ne connait
ni pandas ni les fichiers. Il est donc testable seul et reutilisable
partout.

Contient :
  - les fonctions statistiques recodees A LA MAIN (count, mean, std,
    minimum, maximum, percentile)
    (describe, np.mean, np.std, np.percentile, etc.) ;
  - les helpers de preparation des features : fill_na_with_mean,
    standardize.

Contrat commun :
  toutes les fonctions prennent une sequence de valeurs (liste, Series,
  colonne...) et ignorent les valeurs manquantes (NaN), exactement comme
  pandas.describe(). std utilise le diviseur (n-1) et percentile
  l'interpolation lineaire, pour matcher pandas au chiffre pres.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Nettoyage : on retire les NaN avant tout calcul (comportement pandas)
# ---------------------------------------------------------------------------

def _clean(values: Iterable) -> list[float]:
    """Retourne une liste de floats sans les valeurs manquantes (NaN)."""
    cleaned: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            cleaned.append(f)
    return cleaned


# ---------------------------------------------------------------------------
# Fonctions statistiques recodees a la main
# ---------------------------------------------------------------------------

def count(values: Iterable) -> float:
    """Nombre de valeurs NON nulles (comme pandas)."""
    n = 0
    for _ in _clean(values):
        n += 1
    return float(n)


def mean(values: Iterable) -> float:
    """Moyenne arithmetique."""
    data = _clean(values)
    if not data:
        return float("nan")
    total = 0.0
    for v in data:
        total += v
    return total / len(data)


def std(values: Iterable) -> float:
    """Ecart-type d'echantillon (diviseur n-1, comme pandas)."""
    data = _clean(values)
    n = len(data)
    if n < 2:
        return float("nan")
    m = mean(data)
    acc = 0.0
    for v in data:
        acc += (v - m) ** 2
    return math.sqrt(acc / (n - 1))


def minimum(values: Iterable) -> float:
    """Valeur minimale."""
    data = _clean(values)
    if not data:
        return float("nan")
    lo = data[0]
    for v in data[1:]:
        if v < lo:
            lo = v
    return lo


def maximum(values: Iterable) -> float:
    """Valeur maximale."""
    data = _clean(values)
    if not data:
        return float("nan")
    hi = data[0]
    for v in data[1:]:
        if v > hi:
            hi = v
    return hi


def percentile(values: Iterable, p: float) -> float:
    """Percentile p (0-100) avec interpolation lineaire (comme pandas/numpy).

    Formule : rang virtuel = (p/100) * (n - 1), puis interpolation lineaire
    entre les deux valeurs triees qui l'encadrent.
    """
    data = sorted(_clean(values))
    n = len(data)
    if n == 0:
        return float("nan")
    if n == 1:
        return data[0]
    rank = (p / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return data[int(rank)]
    weight = rank - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


# ---------------------------------------------------------------------------
# Preparation des features
# ---------------------------------------------------------------------------

def fill_na_with_mean(values: Iterable, m: float | None = None) -> list[float]:
    """Remplace les NaN d'une colonne par la moyenne (imputation simple).

    m peut etre fourni (ex. moyenne calculee sur le train et reutilisee sur
    le test) ; sinon il est calcule a partir des valeurs.
    """
    if m is None:
        m = mean(values)
    result: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        result.append(m if math.isnan(f) else f)
    return result


def standardize(
    values: Iterable,
    m: float | None = None,
    s: float | None = None,
) -> tuple[list[float], float, float]:
    """Standardise une colonne : (x - moyenne) / ecart-type.

    Passer m et s permet d'appliquer AU TEST la transformation calculee sur
    le TRAIN (indispensable pour que les predictions soient coherentes).
    Retourne aussi (m, s) pour pouvoir les sauvegarder dans le fichier
    de poids.
    """
    if m is None:
        m = mean(values)
    if s is None:
        s = std(values)
    if s == 0 or math.isnan(s):
        s = 1.0
    standardized = [(v - m) / s for v in values]
    return standardized, m, s
