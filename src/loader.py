"""
loader.py — Chargement de fichier et manipulation de DataFrame du projet
DSLR.

Regroupe tout ce qui touche a pandas et aux fichiers : lecture du CSV et
selection des colonnes numeriques. C'est le seul module qui depend de
pandas.

pandas est utilise UNIQUEMENT pour lire le CSV et inspecter les types de
colonnes : aucun calcul statistique n'y est fait (ceux-la sont recodes a
la main dans math_utils.py).
"""

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Charge un CSV dans un DataFrame pandas (lecture seule, autorise)."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Erreur : fichier introuvable '{path}'")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        raise SystemExit(f"Erreur de lecture de '{path}' : {err}")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des noms de colonnes numeriques du DataFrame.

    Sert a exclure les colonnes texte (nom, maison, date, main...) des
    calculs statistiques et du modele. Attention : 'Index' est numerique
    mais reste un identifiant, a exclure des features.
    """
    cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols
