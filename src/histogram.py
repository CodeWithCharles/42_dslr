"""
histogram.py — Histogrammes des notes par maison, par cours.

Pour chaque cours (colonne numerique hors 'Index'), trace un histogramme
des notes ventilees par maison de Poudlard (4 series superposees), afin de
repondre a la question du sujet : quel cours a une distribution de notes
homogene entre les 4 maisons ? Le classement d'homogeneite est calcule
avec math_utils.mean/std (aucun nouveau calcul statistique ajoute) ; voir
doc/common.md pour le socle partage et doc/histogram.md pour le
fonctionnement propre a ce script.

Usage : python3 histogram.py <dataset.csv> [--save <fichier.png>]
"""

from __future__ import annotations

import math
import sys

import matplotlib.pyplot as plt

from houses import HOUSE_COLORS, HOUSES, require_house_column, scores_by_house
from loader import load_csv, numeric_columns
from math_utils import mean, relative_std, standardize

BINS: int = 20


def standardized_scores_by_house(df, course: str) -> dict[str, list[float]]:
    """Comme scores_by_house, mais les notes sont standardisees
    (math_utils.standardize) avec la moyenne/ecart-type du cours entier :
    rend les histogrammes comparables entre cours d'echelles differentes."""
    raw = scores_by_house(df, course)
    _, m, s = standardize(df[course].tolist())
    return {house: [(v - m) / s for v in values] for house, values in raw.items()}


def homogeneity_score(df, course: str) -> float:
    """Mesure a quel point les 4 maisons ont des notes centrees pareil sur
    ce cours : ecart-type des 4 moyennes par maison (math_utils.mean),
    normalise par l'ecart-type de toutes les notes du cours
    (math_utils.relative_std) pour rendre le score comparable entre cours
    d'echelles tres differentes (ex. Arithmancy vs Care of Magical
    Creatures). Plus le score est bas, plus la distribution est homogene
    entre maisons."""
    house_means = [mean(values) for values in scores_by_house(df, course).values()]
    return relative_std(house_means, df[course].tolist())


def rank_courses(df, courses: list[str]) -> list[tuple[str, float]]:
    """Classe les cours du plus homogene (score le plus bas) au moins
    homogene."""
    scored = [(course, homogeneity_score(df, course)) for course in courses]
    return sorted(scored, key=lambda item: item[1])


def print_ranking(ranking: list[tuple[str, float]]) -> None:
    """Affiche le classement d'homogeneite et met en avant la reponse."""
    print("Classement d'homogeneite (ecart-type des moyennes par maison / ecart-type du cours, du plus au moins homogene) :")
    for course, score in ranking:
        print(f"  {course:<32} {score:.6f}")
    best_course, _ = ranking[0]
    print(f"\nCours le plus homogene entre les 4 maisons : {best_course}")


def plot_histograms(df, courses: list[str], save_path: str | None = None) -> None:
    """Trace une grille d'histogrammes (un par cours), 4 series superposees
    par maison, et affiche ou sauvegarde la figure. Les notes sont
    standardisees par cours (standardized_scores_by_house) pour que tous
    les subplots partagent la meme echelle (ecart-types), comparable
    d'un cours a l'autre malgre des unites brutes tres differentes."""
    n_cols = 4
    n_rows = math.ceil(len(courses) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten() if len(courses) > 1 else [axes]

    for ax, course in zip(axes_flat, courses):
        for house, values in standardized_scores_by_house(df, course).items():
            if values:
                ax.hist(values, bins=BINS, alpha=0.5, label=house, color=HOUSE_COLORS[house])
        ax.set_title(course, fontsize=9)
        ax.set_xlim(-4, 4)

    for ax in axes_flat[len(courses):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()


def _parse_args(argv: list[str]) -> tuple[str, str | None]:
    """Parse les arguments : chemin du dataset et option --save optionnelle."""
    if len(argv) == 2:
        return argv[1], None
    if len(argv) == 4 and argv[2] == "--save":
        return argv[1], argv[3]
    raise SystemExit("Usage : python3 histogram.py <dataset.csv> [--save <fichier.png>]")


def main() -> None:
    """Point d'entree : python3 histogram.py <dataset.csv> [--save <fichier.png>]."""
    path, save_path = _parse_args(sys.argv)
    df = load_csv(path)
    require_house_column(df)

    courses = [c for c in numeric_columns(df) if c != "Index"]
    if not courses:
        raise SystemExit(
            "Erreur : aucune colonne numerique exploitable dans ce fichier "
            "(hors 'Index')."
        )

    ranking = rank_courses(df, courses)
    print_ranking(ranking)
    plot_histograms(df, courses, save_path)


if __name__ == "__main__":
    main()
