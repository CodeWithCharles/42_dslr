"""
logreg_predict.py — Prediction des maisons avec le modele entraine.

Charge les poids et les parametres de normalisation produits par
logreg_train.py, applique la meme selection de features et la meme
normalisation au dataset de test (means/stds du train, jamais recalcules),
calcule une probabilite par maison (one-vs-all) et retient l'argmax. Ecrit
le resultat dans houses.csv au format 'Index,Hogwarts House'. Voir
doc/common.md pour le socle partage et doc/logreg_predict.md pour le detail.

Usage : python3 logreg_predict.py <dataset_test.csv> <weights.json>
                [-o houses.csv]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from logreg_train import add_bias, hypothesis
from preprocessing import load_data, select_features, transform

def load_weights(path: str) -> dict:
    """Charge le fichier de poids JSON produit par logreg_train.py.

    Retourne le dict {features, houses, means, stds, weights}. Leve
    SystemExit avec un message clair si le fichier est introuvable ou
    illisible."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Erreur : fichier de poids introuvable '{path}'")
    except json.JSONDecodeError as err:
        raise SystemExit(f"Erreur de lecture du fichier de poids '{path}' : {err}")

def predict_houses(
    X: np.ndarray,
    houses: list[str],
    weights: dict[str, list[float]]
) -> list[str]:
    """Predit une maison par etudiant (argmax des probabilites one-vs-all).

    Pour chaque maison, calcule la probabilite via son vecteur de poids, puis
    empile ces probas en une matrice (m, nb_maisons). L'argmax par ligne
    donne l'indice de la maison la plus probable ; on le remappe vers son
    nom via l'ordre `houses` (le meme qu'a l'entrainement)."""
    proba = np.column_stack(
        [hypothesis(X, np.array(weights[house])) for house in houses]
    )
    indices = np.argmax(proba, axis=1)
    return [houses[i] for i in indices]

def write_houses(path: str, predictions: list[str]) -> None:
    """Ecrit houses.csv au format exact 'Index,Hogwarts House', l'index
    partant de 0 et suivant l'ordre du fichier de test."""
    with open(path, "w") as f:
        f.write("Index,Hogwarts House\n")
        for index, house in enumerate(predictions):
            f.write(f"{index},{house}\n")

def main() -> None:
    """Point d'entree : charge le modele, predit et ecrit houses.csv."""
    parser = argparse.ArgumentParser(description="Predit les maisons DSLR.")
    parser.add_argument("dataset", help="chemin vers dataset_test.csv")
    parser.add_argument("weights", help="fichier de poids (weights.json)")
    parser.add_argument(
        "-o", "--output",
        default="houses.csv",
        help="fichier de predictions en sortie"
    )
    args = parser.parse_args()

    model = load_weights(args.weights)
    df = load_data(args.dataset)
    columns, _ = select_features(df, model["features"])
    X = add_bias(np.array(transform(columns, model["means"], model["stds"])))
    predictions = predict_houses(X, model["houses"], model["weights"])
    write_houses(args.output, predictions)
    print(f"{len(predictions)} predictions ecrites dans '{args.output}'.")

if __name__ == "__main__":
    main()