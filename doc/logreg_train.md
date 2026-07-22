# `logreg_train.py`

Entraine le modele de regression logistique multi-classe (one-vs-all) par
descente de gradient, puis sauvegarde les poids et les parametres de
normalisation. Voir [`doc/common.md`](common.md) pour le socle partage et
[`doc/preprocessing.md`](preprocessing.md) pour la preparation des features :
ce document ne detaille que ce qui est propre a l'entrainement.

## Utilisation

```bash
python3 src/logreg_train.py data/dataset_train.csv -o weights.json
python3 src/logreg_train.py data/dataset_train.csv --lr 0.5 --iterations 2000
```

| Argument | Role | Defaut |
|---|---|---|
| `dataset` (positionnel) | chemin du CSV d'entrainement | requis |
| `-o`, `--output` | fichier de poids en sortie | `weights.json` |
| `--lr` | learning rate de la descente | `0.5` |
| `--iterations` | nombre d'iterations | `2000` |

`weights.json` est un fichier genere : il ne doit pas etre commit.

## Briques mathematiques (annexe VIII.1 du sujet)

numpy est autorise ici pour l'algebre lineaire brute (produits matriciels,
`exp`) — jamais pour un raccourci statistique. Toutes les operations sur `X`
supposent que la colonne de biais est deja presente.

- `sigmoid(z) = 1 / (1 + exp(-z))` : ecrase un reel dans `]0, 1[`. `z` est
  borne (`clip` a +-500) avant `exp` pour eviter tout overflow sur des
  scores extremes ; a +-500 la sigmoide vaut deja 0 ou 1 a la precision
  machine, le clip ne change donc aucun resultat.
- `hypothesis(X, theta) = sigmoid(X @ theta)` : vecteur de probabilites, une
  par etudiant.
- `cost(X, y, theta)` : cross-entropy
  `-(1/m) * sum[ y*log(h) + (1-y)*log(1-h) ]`. Sert uniquement a SURVEILLER
  la descente (doit decroitre), pas au calcul du gradient. `clip` a `eps`
  pour eviter `log(0)`.
- `gradient(X, y, theta) = (1/m) * X^T @ (h - y)` : vecteur de la taille de
  `theta`, la pente de la perte selon chaque poids.

## Flux d'execution

1. `preprocessing.load_data` charge le CSV ; la colonne `Hogwarts House` est
   extraite comme labels (erreur claire si absente).
2. `preprocessing.select_features` (auto-detection) + `fit_params` calculent
   les colonnes, les moyennes et les ecarts-types du train.
3. `preprocessing.transform` normalise, puis `add_bias` ajoute la colonne de
   1 en tete : `X` passe de 13 a 14 colonnes.
4. `train_one_vs_all` entraine un classifieur binaire par maison.
5. `save_weights` serialise le tout en JSON.

## One-vs-all

La sigmoide ne sait faire qu'une decision binaire. Pour 4 maisons, on
entraine 4 classifieurs independants : pour chacun, `y` vaut 1 si l'etudiant
appartient a la maison, 0 sinon. `train_one_vs_all` boucle sur l'ordre
canonique `HOUSES` (`["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]`)
et produit un vecteur de poids par maison. La prediction (cote
`logreg_predict.py`) prendra la maison dont le classifieur est le plus sur
(argmax).

## Le biais (`add_bias`)

`add_bias` ajoute une feature artificielle toujours egale a 1 en tete de `X`.
Son poids `theta[0]` est le biais (terme constant du modele) : il permet a la
frontiere de decision de ne pas etre forcee de passer par l'origine. Chaque
vecteur de poids compte donc 14 valeurs : le biais en premier, puis les 13
poids de features dans l'ordre de `features`.

## Descente de gradient

`gradient_descent` repete `iterations` fois la mise a jour
`theta -= lr * gradient(X, y, theta)`. `theta` part de zero. Un `lr` trop
grand fait diverger (le cout remonte), trop petit ralentit la convergence.
Le cout final affiche par maison sert de controle : il doit etre bas.

## Format du fichier de poids

JSON, relu tel quel par `logreg_predict.py` :

```json
{
  "features": ["Arithmancy", "Astronomy", "..."],
  "houses": ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
  "means": ["..."],
  "stds": ["..."],
  "weights": {
    "Gryffindor": ["biais", "w1", "...", "w13"]
  }
}
```

Le fichier contient tout le contrat train -> predict : ensemble et ordre des
features, ordre des classes, parametres de normalisation, et les 4 vecteurs
de poids. Le predict n'a donc rien a recalculer.

## Erreurs specifiques a ce script

| Cas | Message / comportement |
|---|---|
| Colonne `Hogwarts House` absente | `SystemExit("Erreur : colonne 'Hogwarts House' absente du train.")` |

Les erreurs de fichier sont gerees par `loader.load_csv` (voir
`doc/common.md`).

## Limites / extensions futures non implementees

- Hyperparametres (`lr`, iterations) a ajuster empiriquement pour viser
  >= 98% d'accuracy ; pas de recherche automatique implementee.
- Descente batch simple : pas de mini-batch ni de critere d'arret anticipe
  sur la convergence du cout.
- Toutes les features numeriques sont utilisees par defaut ; une selection
  affinee (via le pair plot) reste possible en passant une liste explicite a
  `select_features`.