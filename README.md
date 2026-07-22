# DSLR — Datascience x Logistic Regression

Projet 42. Reconstruire le Choixpeau magique : predire la maison de Poudlard
(`Gryffindor`, `Hufflepuff`, `Ravenclaw`, `Slytherin`) d'un etudiant a partir
de ses notes, via une regression logistique multi-classe (one-vs-all)
entrainee par descente de gradient.

Contrainte centrale du sujet : **rien ne doit "faire le travail a notre
place"**. Les statistiques (`count`, `mean`, `std`, `min`, `max`,
`percentile`, ...) et l'entrainement/la prediction sont recodes a la main.
`pandas` sert uniquement a lire les CSV, `numpy` uniquement a l'algebre
lineaire, `sklearn` uniquement a l'auto-evaluation. Detail complet des regles
dans [`AGENTS.md`](AGENTS.md).

## Structure

```
.
├── data/            # datasets fournis (dataset_train.csv, dataset_test.csv)
├── doc/             # documentation : socle, lexique, une page par module
├── src/             # code source
├── AGENTS.md        # regles du projet (regle d'or, architecture, conventions)
├── requirements.txt
└── README.md
```

Modules de `src/` :

| Module | Role |
|---|---|
| `loader.py` | seul point de contact avec pandas/fichiers (lecture CSV, colonnes numeriques) |
| `math_utils.py` | statistiques et helpers recodes a la main (stdlib uniquement) |
| `houses.py` | constantes et helpers partages relatifs aux maisons |
| `preprocessing.py` | preparation des features partagee train/predict (fit/transform) |
| `describe.py` | reimplementation manuelle de `describe()` |
| `histogram.py` | histogrammes des notes par maison, par cours |
| `scatter_plot.py` | nuage de points des deux cours les plus correles |
| `pair_plot.py` | matrice de nuages de points (a venir) |
| `logreg_train.py` | entrainement one-vs-all, export des poids |
| `logreg_predict.py` | prediction des maisons, generation de `houses.csv` |

## Installation

```bash
python3 -m venv venv          # si le venv n'existe pas deja
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Les commandes se lancent depuis la racine du projet.

### Analyse et visualisation

```bash
python3 src/describe.py data/dataset_train.csv
python3 src/histogram.py data/dataset_train.csv [--save histogram.png]
python3 src/scatter_plot.py data/dataset_train.csv [--save scatter.png]
```

### Entrainement

```bash
python3 src/logreg_train.py data/dataset_train.csv -o weights.json \
        [--lr 0.5] [--iterations 2000]
```

Produit `weights.json` : liste des features, ordre des maisons, moyennes et
ecarts-types de normalisation, et les 4 vecteurs de poids.

### Prediction

```bash
python3 src/logreg_predict.py data/dataset_test.csv weights.json -o houses.csv
```

Produit `houses.csv` au format `Index,Hogwarts House`, une ligne par etudiant.

## Auto-evaluation

`dataset_test.csv` a une colonne `Hogwarts House` vide (c'est ce qu'on
predit) : l'accuracy se mesure sur un jeu labellise (le train, ou un split de
validation) avec `sklearn.metrics.accuracy_score`, autorise UNIQUEMENT pour
cette verification. Objectif du sujet : >= 98%.

## Fichiers generes

`weights.json`, `houses.csv` et les `.png` sont generes et ne doivent pas
etre commit (voir `.gitignore`).

## Documentation

Chaque module est documente dans `doc/`. Points d'entree :
[`doc/common.md`](doc/common.md) (socle partage : `loader`, `math_utils`,
`preprocessing`) et [`doc/lexique.md`](doc/lexique.md) (vocabulaire stats/ML).