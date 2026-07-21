# `houses.py`

Module partage (aucun calcul statistique dedans) qui factorise tout ce qui
est commun aux scripts de visualisation qui colorent leurs graphiques par
maison de Poudlard : `histogram.py` et `scatter_plot.py` aujourd'hui,
`pair_plot.py` plus tard. Voir [`doc/common.md`](common.md) pour le socle
partage general (`loader.py`, `math_utils.py`).

## Contenu

- `HOUSE_COLUMN = "Hogwarts House"` : nom de la colonne maison dans les
  CSV du sujet.
- `HOUSES` : tuple des 4 maisons, dans un ordre fixe (stable entre les
  scripts, utile pour que les couleurs/legendes correspondent partout).
- `HOUSE_COLORS` : couleur par maison, inspiree des couleurs officielles
  (rouge Gryffondor, jaune Poufsouffle, bleu Serdaigle, vert Serpentard).
- `require_house_column(df) -> None` : leve `SystemExit` si la colonne
  `Hogwarts House` est absente du DataFrame ou entierement vide (cas de
  `dataset_test.csv`, ou les maisons sont a predire, pas connues). Chaque
  script qui colore par maison appelle cette fonction avant de continuer.
- `scores_by_house(df, course) -> dict[str, list[float]]` : notes non
  manquantes d'un cours, ventilees par maison. Utilise par
  `histogram.py`.
- `paired_scores_by_house(df, course_x, course_y) -> dict[str, tuple[list[float], list[float]]]` :
  paires `(x, y)` de deux cours ou aucune des deux valeurs n'est
  manquante, ventilees par maison. Utilise par `scatter_plot.py` : une
  note NaN sur un seul des deux axes rend la paire inutilisable pour un
  nuage de points (a la difference de `scores_by_house`, qui ne traite
  qu'un seul axe a la fois).

Ce module ne depend que de `math` (pour `math.isnan`) et ne touche ni
pandas ni les fichiers directement (il recoit un DataFrame deja charge par
`loader.load_csv`).
