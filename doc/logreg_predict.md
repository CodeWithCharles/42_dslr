# `logreg_predict.py`

Predit la maison de Poudlard de chaque etudiant de `dataset_test.csv` a
partir du modele entraine par `logreg_train.py`, et ecrit le resultat dans
`houses.csv`. Voir [`doc/common.md`](common.md) pour le socle partage,
[`doc/preprocessing.md`](preprocessing.md) pour la preparation des features
et [`doc/logreg_train.md`](logreg_train.md) pour l'entrainement et le format
du fichier de poids : ce document ne detaille que ce qui est propre a la
prediction.

## Utilisation

```bash
python3 src/logreg_predict.py data/dataset_test.csv weights.json
python3 src/logreg_predict.py data/dataset_test.csv weights.json -o houses.csv
```

| Argument | Role | Defaut |
|---|---|---|
| `dataset` (positionnel) | chemin du CSV a predire | requis |
| `weights` (positionnel) | fichier de poids produit par `logreg_train.py` | requis |
| `-o`, `--output` | fichier de predictions en sortie | `houses.csv` |

`houses.csv` est un fichier genere : il ne doit pas etre commit.

## Flux d'execution

1. `load_weights` lit le fichier JSON de poids et renvoie le dict
   `{features, houses, means, stds, weights}`.
2. `preprocessing.load_data` charge le CSV de test.
3. `preprocessing.select_features(df, model["features"])` extrait les colonnes
   en reutilisant la liste de features SAUVEGARDEE (jamais l'auto-detection),
   pour garantir le meme ensemble et le meme ordre qu'a l'entrainement.
4. `preprocessing.transform(columns, model["means"], model["stds"])` normalise
   avec les parametres du train, puis `add_bias` (importe de `logreg_train`)
   ajoute la colonne de biais.
5. `predict_houses` calcule une probabilite par maison et retient l'argmax.
6. `write_houses` ecrit `houses.csv` au format impose.

## Reutilisation stricte des parametres du train

Le point critique de la prediction : la selection de features et la
normalisation doivent etre IDENTIQUES a celles du train. Le script ne
recalcule donc rien a partir du test :

- l'ensemble et l'ordre des features viennent de `model["features"]` ;
- les moyennes et ecarts-types de standardisation viennent de `model["means"]`
  et `model["stds"]`.

C'est le design fit/transform de `preprocessing.py` qui rend cette contrainte
structurelle : `transform` ne fait qu'appliquer des parametres fournis, il
n'en calcule aucun (voir [`doc/preprocessing.md`](preprocessing.md)).

## Decision one-vs-all : l'argmax

Chaque maison a son propre classifieur binaire (voir
[`doc/logreg_train.md`](logreg_train.md)). `predict_houses` calcule, pour
chaque maison, la probabilite `hypothesis(X, theta_maison)`, puis empile ces
probabilites en une matrice `(nb_etudiants, nb_maisons)` (`np.column_stack`).
`np.argmax(proba, axis=1)` donne, pour chaque etudiant, l'indice de la maison
la plus probable ; cet indice est remappe vers son nom via l'ordre
`model["houses"]` (le meme qu'a l'entrainement). `argmax` n'est pas une
statistique a recoder a la main : c'est une simple recherche d'indice maximum,
usage numpy legitime.

## Format de `houses.csv`

Le sujet impose le format a la lettre, header inclus :

```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
...
```

`write_houses` ecrit le fichier a la main (pas de pandas) pour un controle
exact : une ligne d'en-tete, puis une ligne par etudiant, l'`Index` partant
de 0 et suivant l'ordre du fichier de test. Le nombre de lignes de donnees
egale le nombre d'etudiants du test (pas de ligne supprimee : les NaN sont
imputes, jamais droppes, sinon `houses.csv` n'aurait pas le bon nombre
d'index).

## Erreurs specifiques a ce script

| Cas | Message / comportement |
|---|---|
| Fichier de poids introuvable | `SystemExit("Erreur : fichier de poids introuvable '<path>'")` |
| Fichier de poids illisible (JSON invalide) | `SystemExit("Erreur de lecture du fichier de poids '<path>' : ...")` |

Les erreurs sur le CSV de test (introuvable, illisible) sont gerees par
`loader.load_csv` (voir `doc/common.md`).

## Auto-evaluation (verification seulement)

`dataset_test.csv` a une colonne `Hogwarts House` vide : impossible de mesurer
l'accuracy dessus en local. Pour verifier le modele, on predit sur un jeu
dont on connait les maisons (le train, ou mieux un split de validation) et on
compare avec `sklearn.metrics.accuracy_score` (autorise UNIQUEMENT pour cette
verification, jamais dans le modele). Objectif du sujet : >= 98%.

## Limites / extensions futures non implementees

- Pas de split train/validation integre : l'auto-evaluation se fait a la main.
- L'`Index` ecrit est un compteur (0..n-1) suivant l'ordre du fichier ; il
  n'est pas relu depuis la colonne `Index` du test (les deux coincident sur
  les datasets fournis).