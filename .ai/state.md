# État du projet DSLR — Partie B (Logistic Regression)

## Fait
- `src/math_utils.py` — stats maison (count, mean, std, min, max, percentile) +
  `fill_na_with_mean(values, m=None)` + `standardize(values, m=None, s=None)`.
  Les deux helpers acceptent m/s pré-calculés → réutilisables au predict.
- `src/loader.py` — `load_csv(path)`, `numeric_columns(df)`.

- `src/preprocessing.py` — module partagé train/predict. Fait & validé
  (13 features auto-détectées, 1600 lignes, standardisation OK). Design
  fit/transform : `fit_params` apprend means/stds (train), `transform` les
  applique (train+predict), `select_features(df, None)` auto-détecte et rend
  la liste résolue à sauvegarder.

## En cours
- Briques math de `logreg_train.py` : sigmoid → hypothesis → cost → gradient.

## À faire
- Boucle de descente + one-vs-all + export poids, puis `logreg_predict.py`,
  puis tuning ≥ 98 %.

## Datasets
- `data/dataset_train.csv` (avec Hogwarts House), `data/dataset_test.csv` (House vide).
- 4 maisons : Gryffindor, Hufflepuff, Ravenclaw, Slytherin.
- 13 features de cours (Index + colonnes texte exclus).
