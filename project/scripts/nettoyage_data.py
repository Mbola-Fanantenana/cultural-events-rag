"""
Script de nettoyage et structuration des données.

Utilise les modules de `src/` (notamment `DataCleaner`) pour charger les
données brutes collectées, les nettoyer/structurer, produire des
statistiques et sauvegarder les résultats en vue de l'étape de
vectorisation.

Usage:
    python scripts/nettoyer_dadta.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.data_cleaner import DataCleaner
from config.settings import DATA_RAW_PATH, DATA_PROCESSED_PATH

logger = logging.getLogger(__name__)


def charger_donnees_brutes() -> Optional[List[Dict[str, Any]]]:
    """
    Charge les données brutes depuis un fichier JSON.

    Cette fonction vérifie l'existence du fichier source contenant les
    événements, puis charge son contenu en mémoire.

    Returns:
        list[dict] | None:
            - Liste des événements si le fichier est trouvé et chargé correctement.
            - None si le fichier n'existe pas, ou si son contenu n'est pas un
              JSON valide, ou n'est pas une liste d'événements.

    Side Effects:
        - Affiche des messages d'information dans la console.
    """

    input_file = Path(DATA_RAW_PATH) / 'evenements_bruts.json'

    if not input_file.exists():
        print(f"\n❌ Fichier non trouvé : {input_file}")
        print("\nExécutez d'abord :")
        print("  python scripts/01_collecter_donnees.py")
        return None

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n❌ Fichier JSON invalide : {input_file}")
        print(f"   Détail : {e}")
        return None
    except OSError as e:
        print(f"\n❌ Impossible de lire le fichier : {input_file}")
        print(f"   Détail : {e}")
        return None

    if not isinstance(events, list):
        print(f"\n❌ Contenu inattendu dans {input_file} : une liste d'événements était attendue, "
              f"reçu {type(events).__name__}")
        return None

    print(f"\n✓ {len(events)} événements chargés")
    print(f"  Fichier : {input_file}")
    print(f"  Taille : {input_file.stat().st_size / 1024:.1f} Ko")

    return events


def nettoyer_et_structurer(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Nettoie et structure les données brutes d'événements.

    Cette fonction utilise la classe DataCleaner pour :
    - Extraire les champs clés
    - Nettoyer les données (suppression des doublons, valeurs invalides, etc.)

    Args:
        events (list[dict]): Liste des événements bruts.

    Returns:
        pd.DataFrame:
            DataFrame contenant les données nettoyées et structurées.
            Peut être vide (0 ligne) si aucun événement n'a survécu au
            nettoyage.

    Side Effects:
        - Affiche des statistiques de nettoyage dans la console.
    """

    print("\n" + "=" * 70)
    print("NETTOYAGE ET STRUCTURATION")
    print("=" * 70)

    cleaner = DataCleaner(events)

    # Extraire les champs clés
    print("\n📋 Extraction des champs clés...")
    df = cleaner.extraire_champs_cles()
    print(f"✓ {len(df)} événements extraits")

    if len(df) == 0:
        print("\n⚠️  Aucun événement exploitable après extraction (champs manquants ou événements malformés).")
        return df

    # Nettoyer
    print("\n🧹 Nettoyage des données...")
    nb_avant = len(df)
    df = cleaner.nettoyer_donnees()
    nb_apres = len(df)
    nb_supprimes = nb_avant - nb_apres

    print("\n✓ Nettoyage terminé")
    print(f"  Événements avant : {nb_avant}")
    print(f"  Événements après : {nb_apres}")
    print(f"  Supprimés : {nb_supprimes} ({nb_supprimes / nb_avant * 100:.1f}%)")

    return df


def _taux_remplissage(series: pd.Series) -> float:
    """
    Calcule le taux (%) de valeurs réellement renseignées dans une série.

    Contrairement à un simple `notna()`, une chaîne vide ou composée
    uniquement d'espaces est considérée comme non renseignée.

    Args:
        series (pd.Series): Série à analyser (attendue de type texte).

    Returns:
        float: Pourcentage (0-100) de valeurs non vides. 0.0 si la série
        est vide.
    """
    if len(series) == 0:
        return 0.0
    rempli = series.notna() & (series.astype(str).str.strip() != '')
    return round(rempli.sum() / len(series) * 100, 1)


def analyser_donnees(df: pd.DataFrame) -> None:
    """
    Analyse les données nettoyées et affiche des statistiques descriptives.

    Les analyses incluent :
    - Top des villes
    - Statistiques sur les longueurs de texte
    - Complétude des champs (une chaîne vide compte comme non renseignée)
    - Présence de catégories, dates et URLs

    Args:
        df (pd.DataFrame): Données nettoyées. Ne doit pas être vide.

    Returns:
        None

    Side Effects:
        - Affiche les résultats dans la console.
    """
    if len(df) == 0:
        print("\n⚠️  Aucune donnée à analyser (DataFrame vide).")
        return

    print("\n" + "=" * 70)
    print("ANALYSE DES DONNÉES")
    print("=" * 70)

    # Top villes
    print("\n📍 Top 10 des villes :")
    top_villes = df['ville'].value_counts().head(10)
    for i, (ville, count) in enumerate(top_villes.items(), 1):
        print(f"   {i:2}. {ville:30} : {count:3} événements")

    # Longueur des descriptions
    print("\n📝 Longueur des textes :")
    desc_lengths = df['description'].str.len()
    print(f"   Description min    : {desc_lengths.min():4.0f} caractères")
    print(f"   Description max    : {desc_lengths.max():4.0f} caractères")
    print(f"   Description moyenne: {desc_lengths.mean():4.0f} caractères")

    # Longueur du texte complet
    if 'texte_complet' in df.columns:
        text_lengths = df['texte_complet'].str.len()
        print(f"   Texte complet min    : {text_lengths.min():4.0f} caractères")
        print(f"   Texte complet max    : {text_lengths.max():4.0f} caractères")
        print(f"   Texte complet moyenne: {text_lengths.mean():4.0f} caractères")

    # Complétude des champs (une chaîne vide compte comme non renseignée)
    print("\n📊 Complétude des champs :")
    for col in df.columns:
        taux = _taux_remplissage(df[col])
        barre = "█" * int(taux / 5)
        print(f"   {col:20} : {barre:20} {taux:5.1f}%")

    # Catégories
    if 'categories' in df.columns:
        taux_cat = _taux_remplissage(df['categories'])
        print(f"\n🏷️  Événements avec catégories : {taux_cat:.1f}%")

    # Dates
    if 'date_debut' in df.columns:
        taux_date = _taux_remplissage(df['date_debut'])
        print(f"📅 Événements avec date : {taux_date:.1f}%")

    # URLs
    if 'url' in df.columns:
        taux_url = _taux_remplissage(df['url'])
        print(f"🔗 Événements avec URL : {taux_url:.1f}%")


def afficher_exemples(df: pd.DataFrame, n: int = 3) -> None:
    """
    Affiche un échantillon d'événements pour inspection.

    Args:
        df (pd.DataFrame): Données nettoyées.
        n (int, optional): Nombre d'exemples à afficher. Défaut = 3.
            Si `n <= 0`, aucun exemple n'est affiché.

    Returns:
        None

    Side Effects:
        - Affiche les événements dans la console.
    """
    if len(df) == 0 or n <= 0:
        return

    print("\n" + "=" * 70)
    print(f"EXEMPLES D'ÉVÉNEMENTS (premiers {min(n, len(df))})")
    print("=" * 70)

    for i in range(min(n, len(df))):
        event = df.iloc[i]
        print(f"\n{'─' * 70}")
        print(f"Événement #{i + 1}")
        print(f"{'─' * 70}")
        print(f"ID          : {event['id']}")
        print(f"Titre       : {event['titre']}")
        print(f"Ville       : {event['ville']}")
        print(f"Date début  : {event['date_debut']}")
        print(f"Date fin    : {event['date_fin']}")
        print(f"URL         : {event['url']}")
        print(f"Description : {event['description'][:100]}...")
        if event['categories']:
            print(f"Catégories  : {event['categories']}")


def sauvegarder_donnees_nettoyees(df: pd.DataFrame) -> bool:
    """
    Sauvegarde les données nettoyées dans différents formats.

    Les fichiers générés :
    - CSV
    - JSON
    - Statistiques détaillées (JSON)

    Args:
        df (pd.DataFrame): Données nettoyées. Ne doit pas être vide.

    Returns:
        bool: True si la sauvegarde s'est déroulée sans erreur, False sinon.

    Side Effects:
        - Crée des fichiers sur le disque.
        - Affiche les informations de sauvegarde.
    """
    if len(df) == 0:
        print("\n⚠️  Rien à sauvegarder (DataFrame vide).")
        return False

    print("\n" + "=" * 70)
    print("SAUVEGARDE DES DONNÉES")
    print("=" * 70)

    try:
        # Créer le dossier
        Path(DATA_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)

        # CSV
        csv_file = Path(DATA_PROCESSED_PATH) / 'evenements_nettoyes.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n✓ CSV sauvegardé : {csv_file}")
        print(f"  Taille : {csv_file.stat().st_size / 1024:.1f} Ko")

        # JSON
        json_file = Path(DATA_PROCESSED_PATH) / 'evenements_nettoyes.json'
        df.to_json(json_file, orient='records', force_ascii=False, indent=2)
        print(f"✓ JSON sauvegardé : {json_file}")
        print(f"  Taille : {json_file.stat().st_size / 1024:.1f} Ko")

        # Statistiques détaillées
        stats: Dict[str, Any] = {
            'nombre_evenements': len(df),
            'colonnes': list(df.columns),
            'villes_principales': df['ville'].value_counts().head(20).to_dict(),
            'completude': {
                col: f"{_taux_remplissage(df[col]):.1f}%"
                for col in df.columns
            },
            'statistiques_texte': {
                'description_min': int(df['description'].str.len().min()),
                'description_max': int(df['description'].str.len().max()),
                'description_moyenne': int(df['description'].str.len().mean()),
            }
        }

        if 'texte_complet' in df.columns:
            stats['statistiques_texte']['texte_complet_min'] = int(df['texte_complet'].str.len().min())
            stats['statistiques_texte']['texte_complet_max'] = int(df['texte_complet'].str.len().max())
            stats['statistiques_texte']['texte_complet_moyenne'] = int(df['texte_complet'].str.len().mean())

        if 'date_debut' in df.columns:
            stats['taux_avec_dates'] = f"{_taux_remplissage(df['date_debut']):.1f}%"

        if 'url' in df.columns:
            stats['taux_avec_url'] = f"{_taux_remplissage(df['url']):.1f}%"

        stats_file = Path(DATA_PROCESSED_PATH) / 'statistiques.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ Statistiques sauvegardées : {stats_file}")
        print(f"  Taille : {stats_file.stat().st_size / 1024:.1f} Ko")

    except OSError as e:
        print(f"\n❌ Erreur lors de la sauvegarde : {e}")
        return False

    return True


def main() -> int:
    """
    Point d'entrée principal du script de pré-processing.

    Pipeline exécuté :
    1. Chargement des données brutes
    2. Nettoyage et structuration
    3. Analyse des données
    4. Affichage d'exemples
    5. Sauvegarde des résultats

    Returns:
        int: Code de sortie (0 si succès, 1 en cas d'échec). Permet
        d'utiliser ce script dans un pipeline automatisé (ex: CI, cron).
    """
    # Route les logs internes de DataCleaner (logger.info/.warning) vers la
    # console, pour conserver le niveau de détail affiché auparavant.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "🧹 NETTOYAGE DES DONNÉES ".center(70, "="))
    print()

    try:
        # Charger
        events = charger_donnees_brutes()
        if events is None:
            return 1

        # Nettoyer
        df = nettoyer_et_structurer(events)

        if len(df) == 0:
            print("\n❌ Aucune donnée après nettoyage !")
            return 1

        # Analyser
        analyser_donnees(df)

        # Afficher des exemples
        afficher_exemples(df, n=3)

        # Sauvegarder
        if not sauvegarder_donnees_nettoyees(df):
            return 1

    except Exception:
        logger.exception("Échec inattendu du pipeline de nettoyage")
        return 1

    print("\n" + "=" * 70)
    print("✅ NETTOYAGE TERMINÉ")
    print("=" * 70)
    print("\nFichiers générés :")
    print(f"  • {DATA_PROCESSED_PATH}/evenements_nettoyes.csv")
    print(f"  • {DATA_PROCESSED_PATH}/evenements_nettoyes.json")
    print(f"  • {DATA_PROCESSED_PATH}/statistiques.json")
    print("\nProchaine étape :")
    print("  python scripts/creer_embeddings.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())