# `scatter_plot.py`

Trace un nuage de points entre deux cours (colonnes numeriques hors
`Index`), colore par maison de Poudlard, et repond a la question posee par
le sujet : *quelles sont les deux features qui se ressemblent le plus
(les plus similaires) ?* Voir [`doc/common.md`](common.md) pour le socle
partage (`loader.py`, `math_utils.py`, formules, regle d'or du sujet) et
[`doc/houses.md`](houses.md) pour les helpers de maison partages avec
`histogram.py` : ce document ne detaille que ce qui est propre a ce
script.

## Utilisation

```bash
python3 src/scatter_plot.py data/dataset_train.csv
python3 src/scatter_plot.py data/dataset_train.csv --save scatter.png
```

Un argument obligatoire (le chemin du CSV) et une option `--save
<fichier.png>` pour sauvegarder la figure au lieu de l'afficher a l'ecran.
Comme `histogram.py`, ce script n'a de sens que sur un dataset ou les
maisons sont connues (`dataset_train.csv`).

## Flux d'execution

1. `loader.load_csv(path)` charge le CSV.
2. `houses.require_house_column(df)` verifie que `"Hogwarts House"` existe
   et contient au moins une valeur exploitable.
3. `loader.numeric_columns(df)` recupere les colonnes numeriques, `Index`
   retire de la liste.
4. `rank_pairs` calcule la correlation de Pearson de **toutes** les paires
   de cours (`math_utils.pearson_correlation`) et les trie par valeur
   absolue decroissante.
5. `print_ranking` affiche le top 5 des paires les plus correlees sur
   stdout et met en avant la meilleure (reponse a la question du sujet).
6. `plot_scatter` trace le nuage de points de la paire gagnante, colore
   par maison (`houses.paired_scores_by_house`), et l'affiche
   (`plt.show()`) ou la sauvegarde (`--save`).

## Calcul de la correlation

`math_utils.pearson_correlation(x, y)` calcule le coefficient de Pearson
`r = cov(x, y) / (std(x) * std(y))` a la main :

1. Nettoyage aligne des deux colonnes (`_clean_pairs`) : une paire est
   retiree si **au moins une** des deux valeurs est manquante — different
   du nettoyage independant de `math_utils._clean` utilise par
   `describe.py`, car ici les deux valeurs doivent rester associees a la
   meme ligne (le meme eleve) pour que la covariance ait un sens.
2. Covariance = moyenne des produits des ecarts a la moyenne, diviseur
   `n - 1` (coherent avec `std`, diviseur `n - 1`).
3. Division par les ecarts-types des deux colonnes (`math_utils.std`,
   deja recode a la main). Retourne `NaN` si une colonne est constante
   (ecart-type nul).

Aucun raccourci interdit (`numpy.corrcoef`, `pandas.DataFrame.corr()`,
`statistics.correlation`) : uniquement `math_utils.mean`/`math_utils.std`,
deja utilises par `describe.py` et `histogram.py`.

Sur `dataset_train.csv`, la paire la plus correlee est `Astronomy` /
`Defense Against the Dark Arts` (r ~= -1.0, correlation quasi parfaite) —
ces deux cours sont donc redondants pour le modele de regression
logistique (l'un des deux peut etre ecarte des features, voir
`pair_plot.py`).

## Rendu graphique

`plot_scatter` trace un point par eleve, colore selon sa maison
(`houses.HOUSE_COLORS`), pour la paire de cours choisie. Contrairement a
`histogram.py`, les valeurs ne sont pas standardisees : les deux axes du
nuage de points restent dans l'unite d'origine du cours, ce qui n'affecte
pas la lecture visuelle de la correlation entre deux memes cours.

## Erreurs specifiques a ce script

| Cas | Message / comportement |
|---|---|
| Nombre d'arguments invalide | `SystemExit("Usage : python3 scatter_plot.py <dataset.csv> [--save <fichier.png>]")` |
| Colonne `Hogwarts House` absente ou entierement vide | Voir `houses.require_house_column`, meme message que `histogram.py` |
| Moins de deux colonnes numeriques hors `Index` | `SystemExit("Erreur : il faut au moins deux colonnes numeriques exploitables (hors 'Index') pour tracer un nuage de points.")` |

Les erreurs de fichier (introuvable, CSV illisible) sont gerees par
`loader.load_csv`, documentees dans `doc/common.md`.
