"""
houses.py — Constantes et helpers partages relatifs aux maisons de
Poudlard (colonne 'Hogwarts House').

Factorise ce qui est commun aux scripts de visualisation qui colorent
leurs graphiques par maison (histogram.py, scatter_plot.py, et plus tard
pair_plot.py) : noms/couleurs des maisons, extraction des notes par
maison, et verification que la colonne est exploitable dans le dataset
charge.
"""

from __future__ import annotations

import math

HOUSE_COLUMN: str = "Hogwarts House"
HOUSES: tuple[str, ...] = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")
HOUSE_COLORS: dict[str, str] = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}


def require_house_column(df) -> None:
    """Leve SystemExit si la colonne Hogwarts House est absente ou vide
    (verification commune a tout script qui colore par maison)."""
    if HOUSE_COLUMN not in df.columns or df[HOUSE_COLUMN].isna().all():
        raise SystemExit(
            f"Erreur : ce dataset ne contient pas de valeurs exploitables dans "
            f"'{HOUSE_COLUMN}'."
        )


def scores_by_house(df, course: str) -> dict[str, list[float]]:
    """Extrait, pour un cours donne, la liste des notes non manquantes de
    chaque maison."""
    result: dict[str, list[float]] = {}
    for house in HOUSES:
        values = df.loc[df[HOUSE_COLUMN] == house, course].tolist()
        result[house] = [v for v in values if not math.isnan(v)]
    return result


def paired_scores_by_house(
    df, course_x: str, course_y: str
) -> dict[str, tuple[list[float], list[float]]]:
    """Extrait, par maison, les paires (x, y) de deux cours ou aucune des
    deux valeurs n'est manquante (necessaire pour un nuage de points :
    une note NaN sur un seul des deux axes rend la paire inutilisable)."""
    result: dict[str, tuple[list[float], list[float]]] = {}
    for house in HOUSES:
        mask = df[HOUSE_COLUMN] == house
        xs_raw = df.loc[mask, course_x].tolist()
        ys_raw = df.loc[mask, course_y].tolist()
        xs: list[float] = []
        ys: list[float] = []
        for xv, yv in zip(xs_raw, ys_raw):
            if not (math.isnan(xv) or math.isnan(yv)):
                xs.append(xv)
                ys.append(yv)
        result[house] = (xs, ys)
    return result
