# Repartition du travail — DSLR

Base commune deja en place : `src/loader.py` (lecture CSV, colonnes numeriques)
et `src/math_utils.py` (stats recodees a la main : `count`, `mean`, `std`,
`minimum`, `maximum`, `percentile`, + `fill_na_with_mean`, `standardize`).
Ces deux fichiers sont le socle partage, ne pas les casser : toute nouvelle
fonction stat/math s'ajoute a `math_utils.py`, toute nouvelle manipulation de
fichier/DataFrame s'ajoute a `loader.py` (voir `AGENTS.md`).

Objectif commun : >= 98% d'accuracy sur `dataset_test.csv`, verifiee avec
`sklearn.metrics.accuracy_score` (jamais utilise pour entrainer/predire).

Le decoupage suit celui du sujet : une partie "analyse de donnees /
visualisation" (V1) et une partie "regression logistique" (V2). Les deux
parties consomment le meme socle mais n'ont quasiment aucune dependance
bloquante entre elles, donc elles peuvent avancer en parallele.

---

## Personne A — Analyse de donnees & visualisation

Fichiers a creer : `src/describe.py`, `src/histogram.py`,
`src/scatter_plot.py`, `src/pair_plot.py`.

### 1. `describe.py`
- Reproduit `pandas.DataFrame.describe()` a la main, colonne numerique par
  colonne numerique, en s'appuyant uniquement sur `math_utils`
  (`count/mean/std/minimum/maximum/percentile`) — jamais `df.describe()`.
- Utilise `loader.load_csv` + `loader.numeric_columns` pour recuperer les
  colonnes a traiter, en excluant `Index`.
- Affiche un tableau (Feature en colonnes, stats en lignes) au format le
  plus proche possible de la sortie pandas.
- Bonus possible : ajouter des stats en plus (`mode`, `skewness`, `range`,
  ...) tant qu'elles sont recodees a la main dans `math_utils.py`.

### 2. `histogram.py`
- Pour chaque cours (feature numerique hors `Index`), trace un histogramme
  des notes par maison (4 series superposees, une couleur par maison).
- Objectif du sujet : identifier la matiere dont la distribution de notes
  est la plus homogene entre les 4 maisons (reponse a donner/afficher).
- matplotlib/seaborn autorises pour le rendu uniquement ; si un calcul est
  necessaire (ex. binning manuel, aplatissement par maison), ca reste de la
  manipulation de donnees simple, pas un raccourci stat interdit.

### 3. `scatter_plot.py`
- Trace les nuages de points pour identifier les deux features les plus
  correlees (a faire visuellement, ou en calculant une correlation a la main
  si besoin — dans ce cas la fonction de correlation va dans
  `math_utils.py`).

### 4. `pair_plot.py`
- Matrice pair-plot (type `seaborn.pairplot`) de toutes les features,
  colorees par maison, pour choisir a l'oeil les features a garder pour le
  modele (etape cle avant l'entrainement : sert de base a la selection de
  features que Personne B utilisera dans `logreg_train.py`).

### Ce qui est produit pour l'autre partie
- Une liste de features retenues (issue de `pair_plot.py` /
  `scatter_plot.py`) a transmettre a Personne B, ex. sous forme de
  commentaire ou petite section README : "features gardees : Astronomy,
  Herbology, ... ; features droppees (redondantes/peu discriminantes) : ...".

---

## Personne B — Regression logistique (train / predict)

Fichiers a creer : `src/logreg_train.py`, `src/logreg_predict.py`, plus
ajout dans `src/math_utils.py` des fonctions manquantes pour la regression
(a la main, stdlib/numpy brut uniquement) :
- `sigmoid(z)`
- fonction de cout (log-loss / cross-entropy)
- calcul du gradient (produits matriciels via `numpy` bruts, pas de
  raccourci d'optimisation)

### 1. `logreg_train.py`
- Charge `dataset_train.csv` via `loader.py`.
- Impute les valeurs manquantes (`fill_na_with_mean`) et standardise les
  features (`standardize`) — garde le `mean`/`std` calcules ici, ils
  doivent etre reutilises tels quels au moment du predict.
- Entraine un modele **one-vs-all** (4 classifieurs binaires, un par
  maison) par **descente de gradient** codee a la main.
- Sauvegarde dans un fichier de poids (ex. `weights.csv`, non commit — voir
  `.gitignore`) :
  - les poids de chacun des 4 classifieurs,
  - la liste des features utilisees (dans l'ordre),
  - `mean`/`std` de standardisation par feature.
- Bonus possible : proposer plusieurs variantes de descente de gradient
  (batch / stochastique / mini-batch) selectionnables en option CLI.

### 2. `logreg_predict.py`
- Charge le fichier de poids + `dataset_test.csv`.
- Standardise le test avec le `mean`/`std` du train (jamais recalcules sur
  le test — regle stricte du sujet et du `AGENTS.md`).
- Calcule les probabilites one-vs-all et predit la maison avec la plus
  forte probabilite par eleve.
- Genere `houses.csv` au format `Index,Hogwarts House` (non commit).
- Verification en local uniquement avec
  `sklearn.metrics.accuracy_score(y_true, y_pred)` contre
  `dataset_truth.csv` si disponible, ou une portion du train mise de cote.

### Dependance vers Personne A
- Attend la liste de features retenues par Personne A (pair-plot) pour
  decider quelles colonnes entrer dans le modele. En attendant, peut
  demarrer/tester avec toutes les features numeriques (hors `Index`) et
  brancher la liste filtree ensuite — pas bloquant pour commencer le code.

---

## Contrat d'interface entre les deux parties

| Element | Producteur | Consommateur | Format |
|---|---|---|---|
| Liste de features retenues | Personne A | Personne B | liste de noms de colonnes |
| `weights.csv` (poids + mean/std + features) | Personne B (train) | Personne B (predict) | CSV interne au binome, non commit |
| `houses.csv` | Personne B (predict) | correction/sujet | `Index,Hogwarts House` |
| `math_utils.py` | commun | commun | ajouts par Personne B (sigmoid, cout, gradient) ; ne pas toucher aux fonctions deja utilisees par Personne A (`describe.py`) |

## Ordre de travail conseille

1. Les deux demarrent en parallele : A sur `describe.py` (rapide, valide
   que `math_utils`/`loader` sont corrects), B sur `logreg_train.py` avec
   toutes les features en attendant le retour de A.
2. A enchaine `histogram.py` -> `scatter_plot.py` -> `pair_plot.py`, transmet
   la liste de features filtrees a B.
3. B branche la liste de features, finalise `logreg_train.py`, puis
   `logreg_predict.py`, verifie l'accuracy (>= 98%).
4. Relecture croisee rapide : A verifie que B n'a pas introduit de
   `sklearn`/raccourci interdit dans le calcul, B verifie que les
   visualisations de A n'appellent pas de fonction stat pandas/numpy
   interdite.
