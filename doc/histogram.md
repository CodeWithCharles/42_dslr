# `histogram.py`

Trace, pour chaque cours (colonne numerique hors `Index`), un histogramme
des notes ventilees par maison de Poudlard, et repond a la question posee
par le sujet : *quel cours a une distribution de notes homogene entre les
4 maisons ?* Voir [`doc/common.md`](common.md) pour le socle partage
(`loader.py`, `math_utils.py`, formules, regle d'or du sujet) : ce document
ne detaille que ce qui est propre a ce script.

## Utilisation

```bash
python3 src/histogram.py data/dataset_train.csv
python3 src/histogram.py data/dataset_train.csv --save histogrammes.png
```

Un argument obligatoire (le chemin du CSV) et une option `--save
<fichier.png>` pour sauvegarder la figure au lieu de l'afficher a l'ecran
(utile en environnement sans affichage). Ce script n'a de sens que sur un
dataset ou les maisons sont connues (`dataset_train.csv`) : voir plus bas
pour le cas `dataset_test.csv`.

## Flux d'execution

1. `loader.load_csv(path)` charge le CSV (erreurs fichier deja gerees,
   voir `doc/common.md`).
2. Verification que la colonne `"Hogwarts House"` existe et contient au
   moins une valeur exploitable (sinon `SystemExit`, voir plus bas).
3. `loader.numeric_columns(df)` recupere les colonnes numeriques, `Index`
   retire de la liste — comme dans `describe.py`.
4. Pour chaque cours restant, `rank_courses` appelle `homogeneity_score`,
   qui appelle `scores_by_house` (repartition des notes par maison) puis
   `math_utils.mean`/`math_utils.std` (voir plus bas).
5. `print_ranking` affiche le classement complet sur stdout et met en
   avant le cours le plus homogene (reponse a la question du sujet).
6. `plot_histograms` construit la grille de figures et l'affiche
   (`plt.show()`) ou la sauvegarde (`--save`).

Aucun calcul statistique n'a lieu directement dans `histogram.py` en
dehors de la reutilisation de `math_utils.mean`/`math_utils.std` : le
script se contente d'orchestrer `loader.py`, `math_utils.py` et matplotlib.

## Calcul du classement d'homogeneite

Le sujet demande de repondre a la question visuellement, mais un chiffre
objectif evite de trancher a l'oeil sur 13 histogrammes superposes. Pour
chaque cours, `homogeneity_score` :

1. Calcule, pour chacune des 4 maisons, la moyenne de ses notes sur ce
   cours via `math_utils.mean` (ignore deja les NaN).
2. Calcule l'ecart-type de ces 4 moyennes via `math_utils.std`.

Plus cet ecart-type est bas, plus les 4 maisons ont des notes centrees au
meme endroit sur ce cours -> distribution homogene entre maisons. Ce n'est
**pas** une nouvelle fonction statistique cachee : c'est une reutilisation
directe de `mean`/`std`, deja recodes a la main dans `math_utils.py` pour
`describe.py`. Aucun ajout n'a ete fait a `math_utils.py` pour ce script.

Sur `dataset_train.csv`, le cours le plus homogene est
`Care of Magical Creatures` (ecart-type des 4 moyennes ~0.06, tres
inferieur aux autres cours dont l'ecart-type des moyennes depasse souvent
plusieurs unites) — visible aussi sur l'histogramme correspondant, ou les
4 distributions se superposent presque parfaitement.

## Rendu graphique

`plot_histograms` place un subplot par cours dans une grille (4 colonnes),
avec 4 histogrammes superposes par subplot (`alpha=0.5` pour la
transparence, une couleur par maison via `HOUSE_COLORS` — inspiree des
couleurs officielles de chaque maison : rouge Gryffondor, jaune Poufsouffle,
bleu Serdaigle, vert Serpentard). Une legende commune est affichee une
seule fois pour toute la figure. Les subplots excedentaires (grille pas
totalement remplie) sont masques (`ax.axis("off")`).

## Erreurs specifiques a ce script

| Cas | Message / comportement |
|---|---|
| Nombre d'arguments invalide | `SystemExit("Usage : python3 histogram.py <dataset.csv> [--save <fichier.png>]")` |
| Colonne `Hogwarts House` absente ou entierement vide | `SystemExit("Erreur : ce dataset ne contient pas de valeurs exploitables dans 'Hogwarts House' (histogramme par maison impossible).")` — cas de `dataset_test.csv`, ou les maisons sont a predire, pas connues |
| Aucune colonne numerique hors `Index` | `SystemExit("Erreur : aucune colonne numerique exploitable dans ce fichier (hors 'Index').")` |

Les erreurs de fichier (introuvable, CSV illisible) sont gerees par
`loader.load_csv`, documentees dans `doc/common.md`.

## Limites / extensions futures non implementees

Le classement d'homogeneite se base uniquement sur l'ecart-type des
moyennes par maison ; une metrique combinant aussi la dispersion
intra-maison (ex. ecart-type moyen de chaque maison) pourrait affiner le
classement mais n'est pas necessaire pour repondre a la question du sujet
— non implementee ici.
