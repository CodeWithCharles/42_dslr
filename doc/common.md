# Socle commun du projet DSLR

Ce document decrit ce qui est partage par TOUS les scripts du sujet
(`describe.py`, `histogram.py`, `scatter_plot.py`, et plus tard
`pair_plot.py`, `logreg_train.py`, `logreg_predict.py`) : le chargement des
donnees et le calcul des statistiques. Chaque script a sa propre doc dans
`doc/` (ex. `doc/describe.md`, `doc/histogram.md`, `doc/scatter_plot.md`)
pour ce qui lui est specifique ; les helpers de maison partages entre
scripts de visualisation sont dans `houses.py` / `doc/houses.md`. Ce
fichier n'est pas duplique ailleurs.

## Regle d'or du sujet

Rien ne doit "faire le travail a notre place" : `pandas.DataFrame.describe()`,
`numpy.mean/std/percentile`, `statistics.*` et `sklearn` (pour
entrainer/predire) sont interdits partout sauf exceptions ci-dessous. Le
detail complet des contraintes est dans `AGENTS.md` a la racine du repo ;
ce document n'en est qu'un resume applique au socle.

## `src/loader.py` — seul point de contact avec pandas/fichiers

- `load_csv(path: str) -> pd.DataFrame` : lit un CSV avec pandas (lecture
  seule, autorisee), leve `SystemExit` avec un message clair si le fichier
  est introuvable ou illisible.
- `numeric_columns(df: pd.DataFrame) -> list[str]` : renvoie les noms de
  colonnes de dtype numerique (pandas `is_numeric_dtype`). Aucun calcul
  statistique n'y transite : cette fonction ne fait qu'inspecter les types.

Point d'attention : `numeric_columns` renvoie aussi `Index` (numerique par
dtype), et peut renvoyer une colonne entierement vide comme numerique si
pandas l'a inferee en `float64` faute de valeurs (cas typique de
`Hogwarts House` dans `dataset_test.csv`, ou toutes les valeurs sont
manquantes) — chaque script consommateur doit filtrer selon son besoin
(`Index` toujours exclu, voir plus bas).

## `src/math_utils.py` — les stats recodees a la main

Module stdlib pur (`math` uniquement), sans dependance a pandas/numpy. Prend
en entree n'importe quelle sequence de valeurs (liste, `Series`, ...).

Contrat commun a toutes les fonctions :
- Les valeurs manquantes (NaN) sont ignorees avant tout calcul
  (`_clean`), exactement comme le ferait `pandas.describe()`.
- `std` utilise le diviseur `n-1` (correction de Bessel), pas `n`.
- `percentile` utilise l'interpolation lineaire (comme pandas/numpy par
  defaut), pas de methode "nearest" ou "midpoint".

Ces choix ne sont pas arbitraires : ils permettent de retrouver exactement
les memes valeurs que `pandas.DataFrame.describe()` sans jamais l'appeler.

### Fonctions statistiques

- `count(values) -> float` : nombre de valeurs non manquantes.
- `mean(values) -> float` : moyenne arithmetique = somme / nombre de
  valeurs.
- `std(values) -> float` : ecart-type d'echantillon,
  `sqrt(sum((x - mean)^2) / (n - 1))`. Retourne `NaN` si moins de 2 valeurs.
- `minimum(values) -> float` / `maximum(values) -> float` : parcours
  lineaire des valeurs nettoyees.
- `percentile(values, p) -> float` : percentile `p` (0-100). Rang virtuel
  `r = (p/100) * (n-1)`, puis interpolation lineaire entre les deux valeurs
  triees qui encadrent `r`. Si `r` tombe exactement sur une valeur, elle
  est retournee directement (pas d'interpolation inutile).

Toutes retournent `NaN` si la sequence ne contient aucune valeur exploitable.

### Statistiques supplementaires (bonus describe)

Ajoutees pour le bonus "add more fields to describe", recodees a la main comme
le reste (aucune fonction toute faite). Elles suivent le meme contrat (NaN
ignores, `NaN` si trop peu de valeurs) et matchent pandas au chiffre pres :

- `variance(values) -> float` : variance d'echantillon (diviseur `n-1`), soit
  le carre de `std`. `NaN` si moins de 2 valeurs.
- `skewness(values) -> float` : asymetrie ajustee Fisher-Pearson
  (`pandas.skew()`). 0 = symetrique, positif = queue a droite, negatif = queue
  a gauche. Definie pour `n >= 3`.
- `kurtosis(values) -> float` : kurtosis excedentaire non biaisee
  (`pandas.kurt()`). 0 = loi normale, positif = queues lourdes, negatif =
  queues legeres. Definie pour `n >= 4`.

`describe.py` en deduit aussi `Range` (`max - min`) et `IQR` (`75% - 25%`)
directement, sans nouvelle fonction (voir `doc/describe.md`).

- `pearson_correlation(x, y) -> float` : coefficient de correlation de
  Pearson entre deux colonnes alignees, `cov(x, y) / (std(x) * std(y))`.
  Nettoie les paires par NaN alignes (une valeur manquante sur l'un des
  deux axes retire la paire entiere), diviseur `n-1` pour la covariance
  (coherent avec `std`). Utilise par `scatter_plot.py` (voir
  `doc/scatter_plot.md`).
- `relative_std(values, reference) -> float` : ecart-type de `values` divise
  par l'ecart-type de `reference` (`std(values) / std(reference)`). Rend deux
  dispersions comparables meme sur des echelles differentes ; renvoie `NaN` si
  l'ecart-type de reference est nul ou `NaN`. Utilise par `histogram.py` pour
  le score d'homogeneite (voir `doc/histogram.md`).

### Helpers de preparation des features

Orchestres par `preprocessing.py` (voir [`doc/preprocessing.md`](preprocessing.md))
pour l'entrainement et la prediction ; pas utilises par `describe.py` :

- `fill_na_with_mean(values, m=None) -> list[float]` : remplace les NaN par
  une moyenne (celle des `values`, ou une moyenne fournie — utile pour
  reappliquer au test la moyenne calculee sur le train).
- `standardize(values, m=None, s=None) -> tuple[list[float], float, float]` :
  centre-reduit une colonne `(x - m) / s`. Permet, comme `fill_na_with_mean`,
  de reutiliser au test le `m`/`s` calcules sur le train — indispensable
  pour que les predictions restent coherentes avec l'entrainement.

## `src/preprocessing.py` — preparation des features partagee train/predict

Module intermediaire entre `loader`/`math_utils` et les scripts de modele. Il
produit la matrice de features `X` de facon identique a l'entrainement et a la
prediction, avec les memes moyennes/ecarts-types (calcules sur le train,
reutilises tels quels au test). Il separe `fit_params` (apprentissage des
parametres, train uniquement) et `transform` (application, train et test).
Aucun calcul statistique brut n'y est fait : tout est delegue a `math_utils`.
Detail complet dans [`doc/preprocessing.md`](preprocessing.md).

## `Index` : toujours exclu des stats et des features

`Index` est une colonne numerique (dtype `int64`) mais c'est un identifiant
de ligne, pas une donnee statistique. `numeric_columns` ne le filtre pas
automatiquement (il ne fait qu'inspecter les dtypes) : chaque script qui
calcule des stats ou construit des features doit l'exclure explicitement de
la liste avant d'appeler `math_utils`.
