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
- **pair_plot** : matrice construite à la main (matplotlib + helpers de
  houses.py), PAS `seaborn.pairplot` (rendu maison + cohérence avec
  scatter/histogram, et pairplot calcule des KDE = zone grise vs règle d'or).
  argparse (diverge de histogram/scatter en sys.argv — assumé).
- **Piège perf pair_plot (169 axes)** : NE PAS utiliser `fig.tight_layout()`
  (~40s sur grosse grille) → `fig.subplots_adjust(...)`. Supprimer les ticks
  avec `set_xticks([])`/`set_yticks([])` (pas juste les labels) → ~25s de
  gagnées. `rasterized=True` sur les scatters. Résultat : ~70s → ~5s.
  L'affichage interactif reste lent par nature → préférer `--save`.
