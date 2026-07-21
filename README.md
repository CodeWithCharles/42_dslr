sigmoid — écrase n'importe quel réel dans ]0, 1[, donc interprétable comme une probabilité. Le np.clip(z, -500, 500) est le seul ajout par rapport à la formule : sans lui, dès que X·theta devient très négatif, np.exp(-z) explose (RuntimeWarning: overflow). À ±500 la sigmoïde vaut déjà 0 ou 1 à la précision d'un float64, donc clipper ne change aucun résultat — ça évite juste les warnings et les inf.

hypothesis — juste sigmoid(X @ theta). @ est le produit matriciel numpy : X est (m, n) (m étudiants, n poids biais inclus), theta est (n,), le résultat est (m,) = une proba par étudiant. Tout se calcule d'un coup, sans boucle.

cost — la cross-entropy. Elle mesure à quel point les probas prédites collent aux labels y (0/1). Tu ne t'en sers pas pour entraîner : elle sert à vérifier que la descente fonctionne (si elle ne décroît pas à chaque itération, ton learning rate est trop grand ou il y a un bug). Le clip à eps évite log(0) quand une proba touche 0 ou 1.

gradient — le vrai moteur. (1/m)·Xᵀ·(h − y) : h − y est le vecteur des erreurs (m,), Xᵀ est (n, m), le produit donne (n,) = une pente par poids. C'est mathématiquement la dérivée de la cross-entropy (le sigmoid et le log se simplifient joliment pour donner cette forme propre — d'où le choix de cette fonction de coût). La descente fera theta -= lr * gradient.

add_bias — le modèle a besoin d'un terme constant (l'ordonnée à l'origine). L'astuce classique : ajouter une colonne de 1. Ainsi θ₀·1 + θ₁·x₁ + … tombe naturellement dans X·theta, sans traiter le biais à part. C'est pour ça que les briques math n'ont aucun cas spécial.

gradient_descent — la boucle est volontairement bête : iterations fois, theta -= lr * gradient. Chaque pas descend la pente de la cross-entropy. lr (learning rate) dose la taille du pas : trop grand → ça diverge (le coût remonte), trop petit → ça converge trop lentement.

train_one_vs_all — le cœur du multi-classe. On ne sait faire que du binaire (une sigmoïde = "oui/non"). Alors on entraîne 4 classifieurs binaires indépendants : "Gryffindor ou pas", "Hufflepuff ou pas", etc. Le vecteur y est reconstruit à chaque tour (1 pour la maison courante, 0 pour les 3 autres). Au predict, on prendra la maison dont le classifieur est le plus confiant (argmax). Le print du coût final te sert de contrôle : il doit être bas (< 0.1 typiquement) et différent d'une maison à l'autre.

save_weights — un seul objet JSON qui contient tout le contrat train↔predict. features fige l'ordre des colonnes, houses fige l'ordre des classes, means/stds la normalisation. Le predict n'aura rien à recalculer.

main — l'enchaînement complet : charge → extrait labels + features → fit_params (train uniquement !) → transform → add_bias → entraîne → sauvegarde. Note l'ordre : on fit_params avant transform, et on sauvegarde means/stds pour que le predict les repasse.