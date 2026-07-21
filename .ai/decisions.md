# Décisions d'architecture — DSLR Partie B

(journal, on n'efface pas)

## À trancher
- **Features retenues** : par défaut les 13 features numériques (hors Index) pour
  valider la mécanique. Affinage via pair plot ensuite.
- **Stratégie NaN** : remplacement par la moyenne de la colonne (train), réutilisée
  au predict. → `math_utils.fill_na_with_mean`.
- **Persistance poids + normalisation** : format à choisir (CSV lisible recommandé).
- **Hyperparamètres** : lr / itérations à ajuster empiriquement.

## Prises
- **preprocessing** : design fit/transform (vs handle_nan/normalize du plan
  initial) pour empêcher structurellement de recalculer la normalisation sur
  le test. std calculé AVANT le fill des NaN (sinon variance faussée).
- **Features** : auto-détection au train via `numeric_columns` moins
  `EXCLUDED={"Index"}`. La liste résolue est rendue par `select_features` et
  sera sauvegardée avec les poids ; le predict la repasse telle quelle.
- **Emplacement** : tous les modules à la racine de `src/` (imports plats).
- **Biais** : colonne de 1 ajoutée côté logreg (train/predict), pas dans
  preprocessing (qui reste une matrice de features pure).
