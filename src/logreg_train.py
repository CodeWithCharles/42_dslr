"""
logreg_train.py — Entrainement du modele de regression logistique
multi-classe (one-vs-all) par descente de gradient.

Entraine un classifieur binaire par maison sur dataset_train.csv, puis
sauvegarde les poids et les parametres de normalisation (means/stds) dans un
fichier JSON relu par logreg_predict.py. numpy est utilise uniquement pour
l'algebre lineaire (produits matriciels, exp), jamais pour un raccourci
statistique. Voir doc/common.md pour le socle partage, doc/preprocessing.md
pour la preparation des features et doc/logreg_train.md pour le detail.

Usage : python3 logreg_train.py <dataset_train.csv> [-o weights.json]
                [--lr 0.5] [--iterations 2000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from houses import HOUSES, HOUSE_COLUMN, require_house_column
from preprocessing import fit_params, load_data, select_features, transform


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide 1 / (1 + exp(-z)), stable numeriquement.

    On borne z avant exp : sur des scores extremes, exp(-z) deborderait
    (overflow) et polluerait l'entrainement. clip a +-500 est neutre car
    sigmoid y vaut deja 0 ou 1 a la precision machine.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Prediction du modele : sigmoid(X . theta). Renvoie un vecteur de
    probabilites, une par etudiant."""
    return sigmoid(X @ theta)


def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Cross-entropy : -(1/m) * sum[ y*log(h) + (1-y)*log(1-h) ].

    eps evite log(0) = -inf quand une proba vaut exactement 0 ou 1. Sert
    seulement a SURVEILLER la descente (doit decroitre), pas au calcul du
    gradient.
    """
    m = len(y)
    h = hypothesis(X, theta)
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    return float(-(1.0 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient de la cross-entropy : (1/m) * X^T * (h - y).

    Renvoie un vecteur de la meme taille que theta : la pente de la perte
    selon chaque poids. C'est ce que la descente va soustraire.
    """
    m = len(y)
    h = hypothesis(X, theta)
    return (1.0 / m) * (X.T @ (h - y))


def add_bias(X: np.ndarray) -> np.ndarray:
    """Ajoute la colonne de biais (des 1) en tete de X.

    Le biais theta0 est le terme constant du modele. En le codant comme une
    'feature' toujours egale a 1, la formule X.theta l'integre sans cas
    particulier. X passe de (m, n) a (m, n+1)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    lr: float,
    iterations: int,
) -> np.ndarray:
    """Descente de gradient batch : a chaque pas, theta -= lr * gradient.

    On repete `iterations` fois. Chaque pas deplace theta dans la direction
    qui fait le plus decroitre la cross-entropy."""
    for _ in range(iterations):
        theta = theta - lr * gradient(X, y, theta)
    return theta


def train_one_vs_all(
    X: np.ndarray,
    labels: list[str],
    lr: float,
    iterations: int,
) -> dict[str, list[float]]:
    """Entraine un classifieur binaire par maison (one-vs-all).

    Pour chaque maison : y vaut 1 si l'etudiant y appartient, 0 sinon. On
    part de theta = 0 et on descend. Resultat : un vecteur de poids par
    maison, range dans un dict {maison: poids}."""
    weights: dict[str, list[float]] = {}
    for house in HOUSES:
        y = np.array([1.0 if label == house else 0.0 for label in labels])
        theta = np.zeros(X.shape[1])
        theta = gradient_descent(X, y, theta, lr, iterations)
        weights[house] = theta.tolist()
        print(f"\t{house:12s} entraine (cout final {cost(X, y, theta):.4f})")
    return weights


def save_weights(
    path: str,
    features: list[str],
    means: list[float],
    stds: list[float],
    weights: dict[str, list[float]],
) -> None:
    """Serialise tout ce que le predict doit relire, en JSON."""
    model = {
        "features": features,
        "houses": list(HOUSES),
        "means": means,
        "stds": stds,
        "weights": weights,
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def main() -> None:
    """Point d'entree : entraine le modele et sauvegarde les poids."""
    parser = argparse.ArgumentParser(description="Entraine le modele DSLR.")
    parser.add_argument("dataset", help="chemin vers dataset_train.csv")
    parser.add_argument(
        "-o", "--output", default="weights.json",
        help="fichier de poids en sortie"
    )
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument(
        "--iterations", type=int, default=2000, help="nombre d'iterations"
    )
    args = parser.parse_args()

    df = load_data(args.dataset)
    require_house_column(df)
    labels = df[HOUSE_COLUMN].tolist()

    columns, features = select_features(df)
    means, stds = fit_params(columns)
    X = add_bias(np.array(transform(columns, means, stds)))

    print(
        f"Entrainement sur {X.shape[0]} etudiants, {len(features)} features, "
        f"lr={args.lr}, {args.iterations} iterations :"
    )
    weights = train_one_vs_all(X, labels, args.lr, args.iterations)
    save_weights(args.output, features, means, stds, weights)
    print(f"Poids sauvegardes dans '{args.output}'.")


if __name__ == "__main__":
    main()