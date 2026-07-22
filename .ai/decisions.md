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
- **Sélection de features (via pair plot)** : train sur 10 features (constante
  `SELECTED_FEATURES` dans logreg_train.py, passée à `select_features`).
  Retirées : Arithmancy + Care of Magical Creatures (pouvoir discriminant
  ~0.03/0.07, bruit) et Defense Against the Dark Arts (doublon parfait
  d'Astronomy, |r|=1). Gardées : Astronomy, Herbology, Divination, Muggle
  Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Charms,
  Flying. Accuracy train IDENTIQUE à 13 features (0.981875, 29 erreurs) →
  confirme que les 3 retirées n'apportaient rien. Le predict hérite via la
  liste sauvegardée dans weights.json.
- **À surveiller** : accuracy train (0.9819) à peine > 98% et optimiste →
  besoin d'un split train/validation pour un chiffre honnête. Penser à
  régénérer weights.json après tout changement de SELECTED_FEATURES (l'artefact
  ne se met pas à jour tout seul).
