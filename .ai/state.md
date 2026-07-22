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

- `src/logreg_predict.py` — FAIT. Charge weights.json, select_features avec
  la liste sauvegardée, transform (means/stds du train), add_bias, argmax des
  probas one-vs-all, écrit houses.csv (header + index depuis 0). Réutilise
  add_bias/hypothesis de logreg_train (pas de duplication).
- Harmonisation code (ASCII, imports groupés, HOUSES/HOUSE_COLUMN importés de
  houses) faite. Reste discuté : helper `feature_columns` dans loader pour
  dédupliquer l'exclusion d'Index (4 fichiers) — pas encore appliqué.
- Docs `doc/` : preprocessing.md, logreg_train.md, logreg_predict.md,
  lexique.md rédigées (format ASCII du binôme).

- `src/pair_plot.py` — FAIT & optimisé. Matrice N×N (diagonale = histos,
  hors diag = scatters colorés par maison), réutilise houses.py. argparse.
  Perf réglée (~5s, voir decisions.md). Répond à la question du sujet sur la
  sélection de features (le script aide à décider, ne filtre pas).
- Docs : preprocessing.md, logreg_train.md, logreg_predict.md, pair_plot.md,
  lexique.md rédigées. README réécrit (contenu erroné remplacé). common.md :
  ajouts preprocessing + relative_std proposés.

## En cours / À faire
- Vérifier accuracy (sanity sur train faite ; mettre en place un vrai split
  train/validation pour un chiffre honnête).
- Tuning lr/itérations si besoin pour viser ≥ 98 %.
- Appliquer le helper `feature_columns` (après accord binôme).
- Optionnel : isoler une liste de features via le pair plot et la passer à
  select_features (le train prend tout par défaut).
- Vérifier que pair_plot n'est plus « à venir » dans README/common.md.

## Datasets
- `data/dataset_train.csv` (avec Hogwarts House), `data/dataset_test.csv` (House vide).
- 4 maisons : Gryffindor, Hufflepuff, Ravenclaw, Slytherin.
- 13 features de cours (Index + colonnes texte exclus).
