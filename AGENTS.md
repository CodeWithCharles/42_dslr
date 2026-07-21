# AGENTS.md — DSLR (Data Science x Logistic Regression)

Projet 42 : reconstruire le Choixpeau magique avec une regression logistique
multi-classe (one-vs-all) sur les notes des eleves de Poudlard, pour predire
leur maison (`Gryffindor`, `Slytherin`, `Ravenclaw`, `Hufflepuff`).

## Regle d'or : rien ne doit "faire le travail a notre place"

C'est la contrainte du sujet (voir `en.subject.dslr.pdf`), et elle prime sur
toute autre consideration de style :

- Interdit : `pandas.DataFrame.describe()`, `numpy.mean/std/percentile`,
  `statistics.*`, `sklearn` pour l'entrainement/la prediction, ou toute
  fonction qui reimplemente peu ou prou `count/mean/std/min/max/percentile`.
- `pandas` est tolere UNIQUEMENT pour lire le CSV et inspecter les dtypes de
  colonnes (`loader.py`). Aucun calcul statistique n'y transite.
- `numpy` est tolere pour l'algebre lineaire brute (produits matriciels,
  `exp`) dans la regression logistique, jamais pour des raccourcis
  statistiques ou d'optimisation tout faits.
- `sklearn.metrics.accuracy_score` est autorise uniquement pour s'auto-evaluer
  en local (le sujet l'utilise pour la correction) — jamais pour entrainer ou
  predire.
- Avant d'ajouter un appel a une lib externe dans du code de calcul, demande-toi :
  "est-ce que ca remplace une fonction que je suis cense coder a la main ?"
  Si oui, code-la a la main dans `math_utils.py`.

## Architecture (repartition des responsabilites)

- `src/loader.py` — seul module autorise a toucher pandas/fichiers : lecture
  CSV, selection des colonnes numeriques. Aucun calcul stat dedans.
- `src/math_utils.py` — stdlib uniquement (`math`). Toutes les stats recodees
  a la main (`count`, `mean`, `std`, `minimum`, `maximum`, `percentile`) plus
  les helpers de preparation des features (`fill_na_with_mean`,
  `standardize`). Contrat commun : ignore les NaN, `std` en `n-1`,
  `percentile` en interpolation lineaire — pour matcher pandas au chiffre
  pres sans l'utiliser.
- Scripts a la racine de `src/` (a creer), un par executable du sujet :
  - `describe.py` — reproduit `describe()` a la main via `math_utils`.
  - `histogram.py`, `scatter_plot.py`, `pair_plot.py` — visualisation
    (matplotlib/seaborn autorises pour le rendu, jamais pour le calcul).
  - `logreg_train.py` — entraine un modele one-vs-all par descente de
    gradient, sauvegarde les poids (+ moyenne/ecart-type de standardisation)
    dans un fichier de poids.
  - `logreg_predict.py` — charge les poids, standardise le test avec les
    memes moyenne/ecart-type que le train (jamais recalcules sur le test),
    genere `houses.csv` au format `Index,Hogwarts House`.

Garde cette separation stricte : nouveau calcul statistique/mathematique →
`math_utils.py` ; nouvelle manipulation de fichier/DataFrame → `loader.py` ;
logique specifique a un executable → son propre script. N'ajoute pas de
dependance pandas/numpy a `math_utils.py`.

## Conventions de code

- Docstrings et commentaires en francais (suit le style existant des fichiers
  actuels), code (noms de fonctions/variables) en anglais.
- Type hints partout (`from __future__ import annotations`, style
  `list[float]`, `tuple[...]`).
- Erreurs utilisateur (fichier introuvable, CSV invalide) -> `SystemExit` avec
  un message clair, pas de stack trace brute.
- `Index` est une colonne numerique mais un identifiant : toujours l'exclure
  des features utilisees par le modele ou les stats descriptives par colonne.

## Workflow train/predict

- Le train calcule ses propres moyenne/ecart-type de standardisation et les
  persiste avec les poids ; le predict reutilise ces valeurs telles quelles
  sur `dataset_test.csv` (jamais de nouveau `mean`/`std` calcules sur le test).
- Objectif de performance : >= 98% d'accuracy sur `dataset_test.csv` (mesure
  avec `sklearn.metrics.accuracy_score`, en verification seulement).

## Environnement

- `.venv/` deja present, dependances dans `requirements.txt` (pandas, numpy,
  matplotlib, seaborn). Utiliser `python3` de ce venv pour tout run/test.
- Les CSV de `data/` sont trackes malgre le `.gitignore` global `*.csv` (voir
  les exceptions `!datasets/dataset_train.csv` / `!datasets/dataset_test.csv`)
  — ne jamais commit de nouveaux CSV generes (`houses.csv`, `weights.csv`)
  sans verifier qu'ils restent ignores.
