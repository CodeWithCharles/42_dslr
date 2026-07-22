# `pair_plot.py`

Affiche une matrice de nuages de points (pair plot) : pour chaque paire de
cours numeriques (hors 'Index'), un nuage colore par maison ; la diagonale
montre l'histogramme de chaque cours par maison. Repond a la question du
sujet : quelles features retenir pour la regression logistique ? Voir
[`doc/common.md`](common.md) pour le socle partage, [`doc/houses.md`](houses.md)
pour les helpers de maison reutilises, et [`doc/scatter_plot.md`](scatter_plot.md)
pour la correlation de Pearson (complementaire a la lecture visuelle).

## Utilisation

```bash
python3 src/pair_plot.py data/dataset_train.csv
python3 src/pair_plot.py data/dataset_train.csv --save pair_plot.png
```

| Argument | Role | Defaut |
|---|---|---|
| `dataset` (positionnel) | chemin du CSV a visualiser | requis |
| `-s`, `--save` | sauvegarde la figure dans un fichier au lieu de l'afficher | (affichage) |

Ce script utilise `argparse` (contrairement a `histogram.py`/`scatter_plot.py`
qui parsent a la main) : `-h` liste l'aide automatiquement.

## Flux d'execution

1. `loader.load_csv` charge le CSV, `require_house_column` verifie la colonne
   de maison (indispensable pour colorer).
2. Les cours sont les colonnes numeriques hors 'Index'.
3. `plot_pair_plot` construit la grille N x N : diagonale via `_draw_diagonal`
   (histogrammes), hors diagonale via `_draw_scatter` (nuages).
4. Affichage (`plt.show`) ou sauvegarde (`--save`).

Aucun calcul statistique dans ce script : l'extraction des notes par maison
est deleguee a `houses.py` (`scores_by_house`, `paired_scores_by_house`),
matplotlib ne fait que le rendu.

## Structure de la matrice

- **Hors diagonale** (cellule ligne i, colonne j) : nuage de points du cours j
  (axe x) contre le cours i (axe y), une couleur par maison.
  `paired_scores_by_house` ne garde que les paires ou aucune des deux notes
  n'est manquante.
- **Diagonale** (i == j) : un nuage n'aurait aucun sens (x == y), on affiche
  donc l'histogramme du cours, une serie par maison.
- Les ticks sont supprimes (illisibles a cette densite) et seuls les bords
  (colonne de gauche, ligne du bas) portent le nom des cours.

## Repondre a la question du sujet

Le pair plot sert a choisir les features de la regression. On y lit :

- **Paires redondantes** : un nuage en droite = deux cours quasi parfaitement
  correles (ils portent la meme information). En garder un seul.
- **Features non discriminantes** : cellules ou les 4 maisons se superposent
  totalement. Elles ne separent pas les classes, candidates a ecarter.
- **Features discriminantes** : cellules ou les couleurs se separent nettement.
  A garder.

La selection choisie peut ensuite etre passee explicitement a
`preprocessing.select_features(df, features)` cote entrainement (par defaut, le
train utilise toutes les features numeriques).

## Performance

La matrice compte N x N cellules (13 features => 169 sous-graphes). Deux
pieges de rendu, evites dans ce script :

- **Ne pas utiliser `fig.tight_layout()`** : sur 169 axes avec labels, il coute
  des dizaines de secondes. On positionne les cellules avec
  `fig.subplots_adjust(...)` (temps constant).
- **Supprimer les ticks** (`set_xticks([])`/`set_yticks([])`) et non seulement
  leurs labels : evite le calcul des graduations sur chaque cellule.

Les points sont traces en `rasterized=True` (bitmap) pour un PNG leger malgre
~270k points. Le rendu sauvegarde prend quelques secondes ; l'affichage
interactif d'une grille aussi dense reste lent par nature (preferer `--save`).

## Erreurs specifiques a ce script

| Cas | Message / comportement |
|---|---|
| Colonne de maison absente ou vide | `SystemExit` via `houses.require_house_column` |
| Moins de 2 colonnes numeriques (hors 'Index') | `SystemExit("Erreur : il faut au moins deux colonnes numeriques exploitables (hors 'Index') pour tracer un pair plot.")` |

Les erreurs de fichier sont gerees par `loader.load_csv` (voir `doc/common.md`).

## Limites / extensions futures non implementees

- Selection de features non automatisee : le script aide a la decision, il ne
  filtre pas les cours lui-meme.
- Pas de sous-echantillonnage des points ; suffisant pour 1600 lignes, a
  reconsiderer sur un dataset beaucoup plus gros.