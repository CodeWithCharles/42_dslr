"""
describe.py — Reimplementation manuelle de pandas.DataFrame.describe().

Calcule les statistiques descriptives (count, mean, std, min, 25%, 50%,
75%, max) de toutes les colonnes numeriques d'un dataset, hors 'Index'
(identifiant de ligne, pas une feature), plus des champs supplementaires en
bonus (var, range, iqr, skew, kurt). Toutes les statistiques sont calculees
via math_utils.py (aucun calcul pandas/numpy/statistics ici) ; voir
doc/common.md pour le detail du socle partage et doc/describe.md pour le
fonctionnement propre a ce script.

Usage : python3 describe.py <dataset.csv>
"""

from __future__ import annotations

import math
import shutil
import sys

import pandas as pd

from loader import load_csv
from math_utils import (
    count,
    kurtosis,
    maximum,
    mean,
    minimum,
    percentile,
    skewness,
    std,
    variance,
)
from preprocessing import select_features

STAT_LABELS: tuple[str, ...] = (
    "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max",
    "Var", "Range", "IQR", "Skew", "Kurt",
)
LABEL_WIDTH: int = 8
COL_WIDTH: int = 16
NUMBER_DECIMALS: int = 6
DEFAULT_TERMINAL_WIDTH: int = 80


def compute_column_stats(values: list[float]) -> dict[str, float]:
    """Calcule les statistiques describe() (8 du mandatory + 5 en bonus)
    pour une colonne.

    Les 8 premieres reproduisent pandas.describe() au chiffre pres ; les
    5 suivantes (variance, etendue, ecart interquartile, asymetrie,
    kurtosis) sont le bonus 'add more fields'. Range et IQR se deduisent des
    stats deja calculees ; var/skew/kurt sont recodees dans math_utils.
    """
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    lo = minimum(values)
    hi = maximum(values)
    return {
        "Count": count(values),
        "Mean": mean(values),
        "Std": std(values),
        "Min": lo,
        "25%": q1,
        "50%": percentile(values, 50),
        "75%": q3,
        "Max": hi,
        "Var": variance(values),
        "Range": hi - lo,
        "IQR": q3 - q1,
        "Skew": skewness(values),
        "Kurt": kurtosis(values),
    }


def compute_describe(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calcule les stats descriptives de toutes les colonnes numeriques,
    hors 'Index' (identifiant de ligne, jamais une feature statistique)."""
    columns, features = select_features(df)
    if not features:
        raise SystemExit(
            "Erreur : aucune colonne numerique exploitable dans ce fichier "
            "(hors 'Index')."
        )
    return {
        name: compute_column_stats(values)
        for name, values in zip(features, columns)
    }


def _truncate(name: str, width: int) -> str:
    """Tronque un nom de colonne trop long pour tenir dans une colonne fixe."""
    if len(name) <= width:
        return name
    return name[: width - 3] + "..."


def _format_value(value: float) -> str:
    """Formate un flottant en notation fixe, ou 'NaN' si valeur manquante."""
    if math.isnan(value):
        return "NaN"
    return f"{value:.{NUMBER_DECIMALS}f}"


def _chunk_columns(column_names: list[str], terminal_width: int) -> list[list[str]]:
    """Decoupe les colonnes en blocs qui tiennent dans la largeur de terminal
    donnee (repli en blocs empiles quand il y a trop de colonnes)."""
    per_block = max(1, (terminal_width - LABEL_WIDTH) // COL_WIDTH)
    return [column_names[i : i + per_block] for i in range(0, len(column_names), per_block)]


def format_table(
    stats: dict[str, dict[str, float]],
    terminal_width: int | None = None,
) -> str:
    """Construit le texte complet du tableau describe(), en blocs de
    colonnes empiles si necessaire pour rester lisible."""
    width = terminal_width or shutil.get_terminal_size(
        fallback=(DEFAULT_TERMINAL_WIDTH, 24)
    ).columns
    columns = list(stats.keys())
    lines: list[str] = []
    for block in _chunk_columns(columns, width):
        header = " " * LABEL_WIDTH + "".join(
            _truncate(name, COL_WIDTH - 2).rjust(COL_WIDTH) for name in block
        )
        lines.append(header)
        for label in STAT_LABELS:
            row = label.ljust(LABEL_WIDTH)
            row += "".join(_format_value(stats[name][label]).rjust(COL_WIDTH) for name in block)
            lines.append(row)
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def print_describe(stats: dict[str, dict[str, float]]) -> None:
    """Affiche le tableau describe() sur la sortie standard."""
    print(format_table(stats))


def main() -> None:
    """Point d'entree : python3 describe.py <dataset.csv>."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage : python3 describe.py <dataset.csv>")
    df = load_csv(sys.argv[1])
    stats = compute_describe(df)
    print_describe(stats)


if __name__ == "__main__":
    main()
