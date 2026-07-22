"""
evaluate.py — Estimation honnete de l'accuracy par split train/validation.

Decoupe dataset_train.csv (labellise) en train + validation jamais vue,
entraine sur le train et mesure l'accuracy sur la validation. Peut repeter
l'operation sur plusieurs seeds et agreger (moyenne, min/max, ecart-type) pour
lisser la variance d'un petit set de validation. Reutilise les briques de
preprocessing/logreg_train/logreg_predict, recode l'accuracy et l'agregation
a la main. Outil local : ne modifie aucun fichier de donnees.

Usage : python3 evaluate.py <dataset_train.csv> [--val-size 100] [--seed 42]
                [--runs 1] [--lr 0.5] [--iterations 2000]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random

import numpy as np

from houses import HOUSE_COLUMN, HOUSES, require_house_column
from logreg_predict import predict_houses
from logreg_train import SELECTED_FEATURES, add_bias, train_one_vs_all
from math_utils import maximum, mean, minimum, std
from preprocessing import fit_params, load_data, select_features, transform


def split_indices(
    n: int, val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Melange les indices 0..n-1 (graine fixe = reproductible) et renvoie
    (indices_train, indices_validation)."""
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return indices[val_size:], indices[:val_size]


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Proportion de predictions correctes (recodee a la main)."""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def evaluate_split(
    df, val_size: int, seed: int, lr: float, iterations: int
) -> float:
    """Entraine sur le train et renvoie l'accuracy sur la validation, pour un
    split defini par `seed`. Les couts par maison de l'entrainement sont tus
    (rediriges) pour ne pas polluer la sortie sur plusieurs runs."""
    train_idx, val_idx = split_indices(len(df), val_size, seed)
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    train_labels = df_train[HOUSE_COLUMN].tolist()
    columns, features = select_features(df_train, SELECTED_FEATURES)
    means, stds = fit_params(columns)
    X_train = add_bias(np.array(transform(columns, means, stds)))
    with contextlib.redirect_stdout(io.StringIO()):
        weights = train_one_vs_all(X_train, train_labels, lr, iterations)

    val_true = df_val[HOUSE_COLUMN].tolist()
    val_columns, _ = select_features(df_val, features)
    X_val = add_bias(np.array(transform(val_columns, means, stds)))
    val_pred = predict_houses(X_val, HOUSES, weights)
    return accuracy(val_true, val_pred)


def main() -> None:
    """Point d'entree : lance un ou plusieurs splits et agrege l'accuracy."""
    parser = argparse.ArgumentParser(
        description="Estime l'accuracy par split train/validation."
    )
    parser.add_argument("dataset", help="chemin vers dataset_train.csv (labellise)")
    parser.add_argument(
        "--val-size", type=int, default=100, help="taille du set de validation"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="graine de depart (par defaut : aleatoire a chaque lancement)"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="nombre de splits (seeds successives : seed, seed+1, ...)"
    )
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument(
        "--iterations", type=int, default=2000, help="nombre d'iterations"
    )
    args = parser.parse_args()

    df = load_data(args.dataset)
    require_house_column(df)
    if args.val_size <= 0 or args.val_size >= len(df):
        raise SystemExit(
            "Erreur : --val-size doit etre entre 1 et le nombre de lignes."
        )
    if args.runs <= 0:
        raise SystemExit("Erreur : --runs doit etre >= 1.")

    base_seed = args.seed if args.seed is not None else random.randrange(2**32)
    print(f"seed de depart : {base_seed}")

    accuracies: list[float] = []
    for i in range(args.runs):
        seed = base_seed + i
        acc = evaluate_split(df, args.val_size, seed, args.lr, args.iterations)
        accuracies.append(acc)
        print(f"  seed {seed:4d} : accuracy {acc:.4f}")

    print(f"\n{args.runs} run(s) | train {len(df) - args.val_size} "
          f"| validation {args.val_size}")
    print(f"moyenne    : {mean(accuracies):.4f}")
    if args.runs > 1:
        print(f"min / max  : {minimum(accuracies):.4f} / {maximum(accuracies):.4f}")
        print(f"ecart-type : {std(accuracies):.4f}")


if __name__ == "__main__":
    main()