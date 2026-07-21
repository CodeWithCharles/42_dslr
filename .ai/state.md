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

- `src/logreg_train.py` — FAIT & lancé. Briques math + add_bias +
  gradient_descent + train_one_vs_all + save_weights (JSON) + main/argparse.
  Run OK : coûts finaux 0.04–0.07, `weights.json` généré (format JSON :
  features, houses, means, stds, weights{maison: theta}).
- Hyperparams actuels : lr=0.5, iterations=2000.

## En cours
- `src/logreg_predict.py` : à écrire (charge weights.json, transform, argmax,
  génère houses.csv).

## À faire
- Génération `houses.csv` + tuning accuracy ≥ 98 %.

## Datasets
- `data/dataset_train.csv` (avec Hogwarts House), `data/dataset_test.csv` (House vide).
- 4 maisons : Gryffindor, Hufflepuff, Ravenclaw, Slytherin.
- 13 features de cours (Index + colonnes texte exclus).
