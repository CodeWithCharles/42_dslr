"""
pair_plot.py — Matrice de nuages de points (pair plot) des cours de Poudlard.

Affiche, pour chaque paire de cours numeriques (hors 'Index'), un nuage de
points colore par maison ; la diagonale montre l'histogramme de chaque cours
par maison. Repond a la question du sujet : quelles features retenir pour la
regression logistique ? (les cours redondants, dont les nuages sont des
droites, et les cours qui ne separent pas les maisons, sont a ecarter.)
Le calcul reste a la main (helpers de houses.py) ; matplotlib ne fait que le
rendu. Voir doc/common.md pour le socle partage et doc/pair_plot.md pour le
fonctionnement propre a ce script.

Usage : python3 pair_plot.py <dataset.csv> [--save <fichier.png>]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from houses import (
    HOUSE_COLORS,
    HOUSES,
    paired_scores_by_house,
    require_house_column,
    scores_by_house,
)
from loader import load_csv, numeric_columns
from preprocessing import select_features

BINS: int = 20

def _draw_diagonal(ax, df, course: str) -> None:
    """Diagonale : histogramme des notes d'un cours, une serie par maison."""
    for house, values in scores_by_house(df, course).items():
        if values:
            ax.hist(values, bins=BINS, alpha=0.5, color=HOUSE_COLORS[house])

def _draw_scatter(ax, df, course_x: str, course_y: str) -> None:
    """Hors diagonale : nuage de points de deux cours, colore par maison."""
    for house, (xs, ys) in paired_scores_by_house(df, course_x, course_y).items():
        if xs:
            ax.scatter(
                xs, ys, s=2, alpha=0.35,
                color=HOUSE_COLORS[house], rasterized=True
            )

def plot_pair_plot(
    df, courses: list[str], save_path: str | None = None
) -> None:
    """Trace la matrice N x N : diagonale = histogrammes, hors diagonale =
    nuages de points. Les ticks sont supprimes (illisibles a cette densite)
    et seuls les bords portent le nom des cours."""
    n = len(courses)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))

    for i, course_y in enumerate(courses):
        for j, course_x in enumerate(courses):
            ax = axes[i][j]
            if i == j:
                _draw_diagonal(ax, df, course_x)
            else:
                _draw_scatter(ax, df, course_x, course_y)

            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(course_y, fontsize=6, rotation=0, ha="right")
            if i == n - 1:
                ax.set_xlabel(course_x, fontsize=6, rotation=45, ha="right")

    handles = [Patch(color=HOUSE_COLORS[house], label=house) for house in HOUSES]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle("Pair plot des cours de Poudlard (couleur = maison)")
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.96, bottom=0.07, wspace=0.08, hspace=0.08
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    else:
        plt.show()

def main() -> None:
    """Point d'entree : charge le dataset et affiche (ou sauvegarde) le pair plot."""
    parser = argparse.ArgumentParser(
        description="Affiche le pair plot des cours de Poudlard."
    )
    parser.add_argument("dataset", help="chemin vers le dataset CSV")
    parser.add_argument(
        "-s", "--save",
        metavar="FICHIER",
        help="sauvegarde la figure dans FICHIER au lieu de l'afficher"
    )
    args = parser.parse_args()

    df = load_csv(args.dataset)
    require_house_column(df)

    _, courses = select_features(df)
    if not courses or len(courses) < 2:
        raise SystemExit(
            "Erreur : il faut au moins deux colonnes numeriques exploitables (hors 'Index') pour tracer un pair plot."
        )
    plot_pair_plot(df, courses, args.save)

if __name__ == "__main__":
    main()