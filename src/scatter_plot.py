"""
scatter_plot.py — Nuage de points entre les deux cours les plus correles.

Calcule la correlation de Pearson (math_utils.pearson_correlation, recodee
a la main) entre chaque paire de cours numeriques (hors 'Index'), pour
repondre a la question du sujet : quelles sont les deux features les plus
similaires (les plus correlees) ? Trace le nuage de points de la paire
gagnante, colore par maison. Voir doc/common.md pour le socle partage et
doc/scatter_plot.md pour le fonctionnement propre a ce script.

Usage : python3 scatter_plot.py <dataset.csv> [--save <fichier.png>]
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from houses import HOUSE_COLORS, paired_scores_by_house, require_house_column
from loader import load_csv, numeric_columns
from math_utils import pearson_correlation


def rank_pairs(df, courses: list[str]) -> list[tuple[tuple[str, str], float]]:
    """Classe toutes les paires de cours par correlation de Pearson
    decroissante en valeur absolue (paire la plus correlee en premier)."""
    scored: list[tuple[tuple[str, str], float]] = []
    for i, course_x in enumerate(courses):
        for course_y in courses[i + 1:]:
            corr = pearson_correlation(df[course_x].tolist(), df[course_y].tolist())
            scored.append(((course_x, course_y), corr))
    return sorted(scored, key=lambda item: abs(item[1]), reverse=True)


def print_ranking(ranking: list[tuple[tuple[str, str], float]], top_n: int = 5) -> None:
    """Affiche les paires les plus correlees et met en avant la meilleure."""
    print(f"Top {min(top_n, len(ranking))} des paires de cours les plus correlees (|r| de Pearson) :")
    for (course_x, course_y), corr in ranking[:top_n]:
        print(f"  {course_x:<24} / {course_y:<24} r = {corr:.6f}")
    (best_x, best_y), best_corr = ranking[0]
    print(f"\nPaire la plus correlee (les plus similaires) : {best_x} / {best_y} (r = {best_corr:.6f})")


def plot_scatter(df, course_x: str, course_y: str, save_path: str | None = None) -> None:
    """Trace le nuage de points de deux cours, une couleur par maison."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for house, (xs, ys) in paired_scores_by_house(df, course_x, course_y).items():
        if xs:
            ax.scatter(xs, ys, alpha=0.6, label=house, color=HOUSE_COLORS[house])
    ax.set_xlabel(course_x)
    ax.set_ylabel(course_y)
    ax.set_title(f"{course_x} / {course_y}")
    ax.legend()
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
    raise SystemExit("Usage : python3 scatter_plot.py <dataset.csv> [--save <fichier.png>]")


def main() -> None:
    """Point d'entree : python3 scatter_plot.py <dataset.csv> [--save <fichier.png>]."""
    path, save_path = _parse_args(sys.argv)
    df = load_csv(path)
    require_house_column(df)

    courses = [c for c in numeric_columns(df) if c != "Index"]
    if len(courses) < 2:
        raise SystemExit(
            "Erreur : il faut au moins deux colonnes numeriques exploitables "
            "(hors 'Index') pour tracer un nuage de points."
        )

    ranking = rank_pairs(df, courses)
    print_ranking(ranking)
    (best_x, best_y), _ = ranking[0]
    plot_scatter(df, best_x, best_y, save_path)


if __name__ == "__main__":
    main()
