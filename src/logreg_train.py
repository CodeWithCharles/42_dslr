"""
entrainement one-vs-all par descente de gradient
"""

from __future__ import annotations

import numpy as np
import argparse
import json

import preprocessing

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoïde 1 / (1 + exp(-z)), stable numériquement.

    On borne z avant exp : sur des scores extrêmes, exp(-z) déborderait
    (overflow) et polluerait l'entraînement. clip à ±500 est neutre car
    sigmoid y vaut déjà 0 ou 1 à la précision machine.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Prédiction du modèle : sigmoid(X · theta). Renvoie un vecteur de
    probabilités, une par étudiant."""
    return sigmoid(X @ theta) # @ = multiplication matricielle

def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Cross-entropy : -(1/m) · Σ [ y·log(h) + (1-y)·log(1-h) ].

    eps évite log(0) = -inf quand une proba vaut exactement 0 ou 1.
    Sert seulement à SURVEILLER la descente (doit décroître), pas au calcul
    du gradient.
    """
    m = len(y)
    h = hypothesis(X, theta)
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    return float(-(1.0 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))

def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient de la cross-entropy : (1/m) · Xᵀ · (h - y).

    Renvoie un vecteur de la même taille que theta : la pente de la perte
    selon chaque poids. C'est ce que la descente va soustraire.
    """
    m = len(y)
    h = hypothesis(X, theta)
    return (1.0 / m) * (X.T @ (h - y))

HOUSES: list[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def add_bias(X: np.ndarray) -> np.ndarray:
    """Ajoute la colonne de biais (des 1) en tête de X.

    Le biais θ₀ est le terme constant du modèle. En le codant comme une
    'feature' toujours égale à 1, la formule X·theta l'intègre sans cas
    particulier. X passe de (m, n) à (m, n+1)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    lr: float,
    iterations: int,
) -> np.ndarray:
    """Descente de gradient batch : à chaque pas, theta -= lr · gradient.

    On répète `iterations` fois. Chaque pas déplace theta dans la direction
    qui fait le plus décroître la cross-entropy."""
    for _ in range(iterations):
        theta = theta - lr * gradient(X, y, theta)
    return theta

def train_one_vs_all(
    X: np.ndarray,
    labels: list[str],
    lr: float,
    iterations: int,
) -> dict[str, list[float]]:
    """Entraîne un classifieur binaire par maison (one-vs-all).

    Pour chaque maison : y vaut 1 si l'étudiant y appartient, 0 sinon. On
    part de theta = 0 et on descend. Résultat : un vecteur de poids par
    maison, rangé dans un dict {maison: poids}."""
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
    """Sérialise tout ce que le predict doit relire, en JSON."""
    model = {
        "features": features,
        "houses": HOUSES,
        "means": means,
        "stds": stds,
        "weights": weights
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Entraine le modele DSLR.")
    parser.add_argument("dataset", help="chemin vers dataset_train.csv")
    parser.add_argument("-o", "--output", default="weights.json", help="fichier de poids en sortie")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument("--iterations", type=int, default=2000, help="nombre d'iterations")
    args = parser.parse_args()

    df = preprocessing.load_data(args.dataset)
    if "Hogwarts House" not in df.columns:
        raise SystemExit("Erreur : colonne 'Hogwarts House' absente du train.")
    labels = df["Hogwarts House"].tolist()

    columns, features = preprocessing.select_features(df)
    means, stds = preprocessing.fit_params(columns)
    X = add_bias(np.array(preprocessing.transform(columns, means, stds)))

    print(f"Entrainement sur {X.shape[0]} etudiants, {len(features)} features, lr={args.lr}, {args.iterations} iterations :")
    weights = train_one_vs_all(X, labels, args.lr, args.iterations)
    save_weights(args.output, features, means, stds, weights)
    print(f"Poids sauvegardes dans '{args.output}'.")

if __name__ == "__main__":
    main()