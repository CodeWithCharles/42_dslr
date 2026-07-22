# `preprocessing.py`

Module partage entre l'entrainement (`logreg_train.py`) et la prediction
(`logreg_predict.py`). Il produit la matrice de features `X` de facon
identique des deux cotes, avec les memes moyennes/ecarts-types. Voir
[`doc/common.md`](common.md) pour le socle partage (`loader.py`,
`math_utils.py`, regle d'or du sujet) : ce document ne detaille que ce qui
est propre a ce module.

C'est le point de couplage critique du projet : si le train et le predict ne
preparaient pas `X` exactement pareil, les poids appris seraient desalignes
et les predictions fausses sans erreur visible.

## Principe : fit / transform

Le module separe deux responsabilites, sur le modele de scikit-learn mais
recodees a la main :

- `fit_params` APPREND les parametres de normalisation (moyennes, ecarts-
  types) — appele UNIQUEMENT sur le train.
- `transform` APPLIQUE des parametres fournis (remplissage des NaN puis
  standardisation) — appele sur le train ET sur le test.

`transform` ne calcule jamais de parametre : il ne fait que consommer ceux
qu'on lui passe. Cette contrainte rend structurellement impossible le fait
de recalculer la normalisation sur le test (interdit par le sujet : le test
doit etre normalise avec les stats du train).

## Fonctions

- `load_data(path) -> pd.DataFrame` : passe-plat vers `loader.load_csv`.
  Permet aux scripts train/predict de n'importer que `preprocessing`.
- `select_features(df, features=None) -> tuple[list[list[float]], list[str]]` :
  - `features=None` (train) : derive automatiquement les colonnes numeriques
    via `loader.numeric_columns(df)`, moins l'ensemble `EXCLUDED`
    (`{"Index"}`).
  - `features` fourni (predict) : utilise cette liste telle quelle.
  Retourne les colonnes (une liste de valeurs par feature, NaN possibles) ET
  la liste des noms retenus. Cette liste est sauvegardee avec les poids par
  le train, puis repassee au predict : un seul chemin de verite pour
  l'ensemble et l'ordre des features.
- `fit_params(columns) -> tuple[list[float], list[float]]` : train uniquement.
  Calcule moyenne et ecart-type par colonne via `math_utils.mean`/`std`, sur
  les valeurs brutes (NaN ignores). Important : l'ecart-type est calcule
  AVANT tout remplissage des NaN, sinon les valeurs imputees gonfleraient `n`
  et fausseraient la variance.
- `transform(columns, means, stds) -> list[list[float]]` : pour chaque
  colonne `i`, remplace les NaN par `means[i]` (`fill_na_with_mean`) puis
  standardise avec `(means[i], stds[i])` (`standardize`). Transpose enfin le
  resultat en lignes (une ligne = un etudiant). Le `(m, s)` renvoye par
  `standardize` est ignore : les parametres sont imposes, pas recalcules.

## Orientation colonnes -> lignes

`math_utils` travaille colonne par colonne (une sequence = une feature),
alors que le modele attend `X` oriente lignes (une ligne = un etudiant).
`select_features`, `fit_params` et `transform` manipulent donc des colonnes,
et `transform` transpose (`zip(*colonnes)`) en toute fin pour rendre `X`.

## Ce qui n'est PAS fait ici

- La colonne de biais (colonne de 1) n'est pas ajoutee : elle est specifique
  a la regression et ajoutee cote `logreg_train.py` / `logreg_predict.py`
  (`add_bias`). `preprocessing.py` reste une matrice de features pure.
- Aucune lecture/ecriture de poids : la persistance des `means`/`stds` (avec
  les poids) est geree par les scripts train/predict.

## Utilisation type

Train :

```python
df = preprocessing.load_data(path)
columns, features = preprocessing.select_features(df)   # auto-detection
means, stds = preprocessing.fit_params(columns)
X = preprocessing.transform(columns, means, stds)
# -> sauvegarder features + means + stds avec les poids
```

Predict :

```python
df = preprocessing.load_data(path)
columns, _ = preprocessing.select_features(df, features)  # liste sauvegardee
X = preprocessing.transform(columns, means, stds)         # params sauvegardes
```