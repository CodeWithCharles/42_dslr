# `describe.py`

Reimplementation manuelle de `pandas.DataFrame.describe()` : calcule pour
chaque colonne numerique (hors `Index`) les statistiques `Count`, `Mean`,
`Std`, `Min`, `25%`, `50%`, `75%`, `Max`. Voir [`doc/common.md`](common.md)
pour le socle partage (`loader.py`, `math_utils.py`, formules, regle d'or du
sujet) : ce document ne detaille que ce qui est propre a ce script.

## Utilisation

```bash
python3 src/describe.py data/dataset_train.csv
python3 src/describe.py data/dataset_test.csv
```

Un seul argument attendu : le chemin du CSV a analyser.

## Flux d'execution

1. `loader.load_csv(path)` charge le CSV (erreurs fichier deja gerees ici,
   voir `doc/common.md`).
2. `loader.numeric_columns(df)` recupere les colonnes numeriques, puis
   `compute_describe` retire `"Index"` de la liste.
3. Pour chaque colonne restante, `compute_column_stats` appelle les 8
   fonctions de `math_utils` (voir `doc/common.md` pour les formules) et
   construit un `dict[str, float]` cle par libelle de stat.
4. `format_table` met en forme le resultat en texte ; `print_describe`
   l'affiche.

Aucun calcul statistique n'a lieu dans `describe.py` lui-meme : le script se
contente d'orchestrer `loader.py` et `math_utils.py`.

## Mise en forme du tableau

Le sujet impose de retrouver les memes valeurs que pandas, mais pas le meme
rendu visuel. Avec 13 colonnes de features (certaines avec un nom long,
comme `Defense Against the Dark Arts`), un tableau sur une seule ligne
deviendrait illisible dans un terminal standard. `format_table` gere donc
deux choses :

- **Troncature des en-tetes** (`_truncate`) : un nom de colonne trop long
  pour la largeur de colonne fixe (`COL_WIDTH = 16`) est coupe et suffixe
  par `"..."` (ex. `Defense Aga...`).
- **Decoupage en blocs** (`_chunk_columns`) : les colonnes sont reparties en
  blocs qui tiennent dans la largeur de terminal detectee
  (`shutil.get_terminal_size`, repli a 80 colonnes si non disponible, ex.
  sortie redirigee vers un fichier). Chaque bloc est affiche l'un sous
  l'autre, avec l'en-tete des stats (`Count`, `Mean`, ...) repete pour
  chaque bloc — comme le fait pandas lui-meme quand une table est trop
  large pour le terminal.

Exemple (extrait reel, `dataset_train.csv`, premier bloc) :

```
              Arithmancy       Astronomy       Herbology  Defense Aga...
Count        1566.000000     1568.000000     1567.000000     1569.000000
Mean        49634.570243       39.797131        1.141020       -0.387863
Std         16679.806036      520.298268        5.219682        5.212794
Min        -24370.000000     -966.740546      -10.295663      -10.162119
25%         38511.500000     -489.551387       -4.308182       -5.259095
50%         49013.500000      260.289446        3.469012       -2.589342
75%         60811.250000      524.771949        5.419183        4.904680
Max        104956.000000     1016.211940       11.612895        9.667405
```

Chaque valeur manquante calculee (ex. `Mean`/`Std` d'une colonne sans
aucune valeur exploitable) s'affiche comme `NaN`.

### Cas particulier : colonne entierement vide traitee comme numerique

Sur `dataset_test.csv`, la colonne `Hogwarts House` est entierement vide (le
but du dataset est de la predire) : pandas l'infere en dtype `float64`
faute de valeurs, donc `numeric_columns` la considere comme numerique. Le
script l'affiche avec `Count = 0` et le reste des stats a `NaN` — c'est le
comportement exact de `pandas.DataFrame.describe()` sur ce meme fichier
(verifie manuellement), pas un bug.

## Erreurs specifiques a ce script

| Cas | Message / comportement |
|---|---|
| Nombre d'arguments != 1 | `SystemExit("Usage : python3 describe.py <dataset.csv>")` |
| Aucune colonne numerique hors `Index` | `SystemExit("Erreur : aucune colonne numerique exploitable dans ce fichier (hors 'Index').")` |

Les erreurs de fichier (introuvable, CSV illisible) sont gerees par
`loader.load_csv`, documentees dans `doc/common.md`.

## Limites / extensions futures non implementees

Cette version couvre uniquement le mandatory du sujet (8 statistiques). Des
statistiques supplementaires (mode, skewness, etendue/IQR, etc.) pourraient
etre ajoutees plus tard en bonus, tant qu'elles restent recodees a la main
dans `math_utils.py` — non implementees ici.
