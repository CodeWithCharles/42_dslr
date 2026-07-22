# Lexique du projet DSLR

Definitions courtes des termes de statistiques et de machine learning
utilises dans le projet. Chaque entree renvoie, si utile, vers la doc du
module concerne. Le socle stat est detaille dans [`doc/common.md`](common.md).

## Accuracy

Proportion de predictions correctes sur un jeu de donnees
(`predictions justes / total`). Objectif du sujet : >= 98% sur
`dataset_test.csv`. Mesuree en verification seulement, via
`sklearn.metrics.accuracy_score` (autorise pour l'auto-evaluation, jamais
pour entrainer ou predire).

## Argmax

Indice (ou classe) de la valeur maximale d'une liste. En prediction, on
calcule une probabilite par maison et on prend l'`argmax` : la maison dont
le classifieur est le plus sur. C'est ce qui transforme 4 scores binaires en
une decision multi-classe.

## Biais

Terme constant du modele (`theta[0]`), independant des features. Code comme
le poids d'une colonne artificielle toujours egale a 1 (voir `add_bias`).
Il permet a la frontiere de decision de ne pas etre forcee de passer par
l'origine : c'est le "decalage par defaut" du modele quand toutes les
features valent 0. A ne pas confondre avec le one-vs-all (qui donne 4
vecteurs) : le biais n'est qu'une seule case a l'interieur de chaque vecteur.

## Classifieur binaire

Modele qui repond a une question a deux issues (oui/non). Une sigmoide seule
ne sait faire que ca. Le multi-classe est obtenu en combinant plusieurs
classifieurs binaires (voir One-vs-all).

## Cout (cross-entropy)

Mesure chiffree de l'erreur du modele : `-(1/m) * sum[ y*log(h) +
(1-y)*log(1-h) ]`. Basse = bon, haute = mauvais. Elle punit tres fort une
erreur confiante (log qui file vers l'infini). Sert a SURVEILLER la descente
(le cout doit decroitre a chaque iteration), pas au calcul du gradient.
Detail dans [`doc/logreg_train.md`](logreg_train.md).

## Descente de gradient (gradient descent)

Algorithme d'optimisation : on part de poids nuls et on les corrige par
petits pas dans la direction qui fait le plus decroitre le cout, en repetant
`theta -= lr * gradient`. Methode d'entrainement imposee par le sujet
(partie mandatory).

## Ecart-type (std)

Mesure de dispersion des valeurs autour de la moyenne. Recode a la main dans
`math_utils.std` avec le diviseur `n-1` (correction de Bessel). Utilise pour
la standardisation.

## Feature

Variable d'entree du modele : ici, une note de cours (Arithmancy, Astronomy,
...). Le projet en retient 13 (colonnes numeriques hors `Index`). L'ensemble
et l'ordre des features sont figes et sauvegardes avec les poids.

## fit / transform

Separation des responsabilites de normalisation (voir
[`doc/preprocessing.md`](preprocessing.md)) : `fit_params` APPREND les
moyennes/ecarts-types (train uniquement), `transform` les APPLIQUE (train et
test). `transform` ne recalcule jamais de parametre, ce qui garantit que le
test est normalise avec les stats du train.

## Gradient

Vecteur des pentes du cout selon chaque poids : `(1/m) * X^T * (h - y)`. Il
indique dans quelle direction et de combien corriger `theta`. C'est le
moteur de la descente de gradient.

## Hypothese (hypothesis)

Sortie du modele avant decision : `sigmoid(X * theta)`, un vecteur de
probabilites (une par etudiant). Notee `h` dans les formules de cout et de
gradient.

## Imputation

Remplacement des valeurs manquantes (NaN) par une valeur de substitution.
Choix du projet : la moyenne de la colonne, calculee sur le train et
reutilisee au test (`math_utils.fill_na_with_mean`).

## Learning rate (lr)

Taille du pas de la descente de gradient. Trop grand : la descente diverge
(le cout remonte). Trop petit : convergence lente. Hyperparametre a ajuster
empiriquement (defaut du projet : `0.5`).

## Moyenne (mean)

Somme des valeurs divisee par leur nombre. Recodee a la main dans
`math_utils.mean` (NaN ignores). Utilisee pour l'imputation et la
standardisation.

## NaN (Not a Number)

Valeur manquante dans les donnees. Ignoree systematiquement avant tout calcul
statistique (`_clean` dans `math_utils`), puis imputee avant l'entrainement
(voir Imputation).

## Normalisation / Standardisation

Mise a la meme echelle des features : `(x - moyenne) / ecart-type`
(`math_utils.standardize`). Chaque feature se retrouve centree sur 0 avec une
dispersion comparable, ce qui rend la descente de gradient stable et rapide.
Les moyennes/ecarts-types sont ceux du train, reutilises tels quels au test.

## One-vs-all

Strategie multi-classe : on entraine un classifieur binaire par classe
("cette maison ou pas"). Pour 4 maisons, 4 vecteurs de poids independants. En
prediction, l'`argmax` des 4 probabilites donne la maison. Detail dans
[`doc/logreg_train.md`](logreg_train.md).

## Percentile

Valeur en dessous de laquelle tombe un pourcentage donne des donnees (ex.
25%, 50%, 75%). Recode a la main dans `math_utils.percentile` avec
interpolation lineaire. Utilise par `describe.py`.

## Poids (theta)

Parametres appris par le modele. Un vecteur par maison, de 14 valeurs : le
biais (`theta[0]`) puis un poids par feature. C'est ce que
`logreg_train.py` optimise et sauvegarde dans `weights.json`.

## Regression logistique

Modele de classification qui predit une probabilite d'appartenance a une
classe via la sigmoide. Combinee au one-vs-all, elle traite les 4 maisons.
C'est le coeur de la partie B du projet.

## Sigmoide

Fonction `1 / (1 + exp(-z))` qui ecrase n'importe quel reel dans `]0, 1[`,
interpretable comme une probabilite. Bornee (`clip` a +-500) avant `exp`
pour eviter les overflows. Base de l'hypothese.