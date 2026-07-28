"""
Script de génération des embeddings vectoriels via l'API Mistral AI.

Ce script constitue la troisième étape du pipeline RAG. Il génère les
embeddings des événements nettoyés et les fusionne avec les embeddings
existants en ne traitant que les nouveaux événements.

Les étapes réalisées sont :
    1. Chargement des événements nettoyés (CSV).
    2. Chargement des embeddings existants (si présents).
    3. Identification des événements sans embedding (nouveaux).
    4. Génération des embeddings manquants via Mistral Embed.
    5. Fusion avec les embeddings existants.
    6. Sauvegarde des fichiers courants et d'un snapshot versionné.

Usage:
    python project/scripts/creer_embeddings.py

Prérequis:
    - Avoir exécuté nettoyage_data.py au préalable.
    - La variable MISTRAL_API_KEY doit être définie dans .env.

Prochaine étape:
    python project/scripts/creer_index_faiss.py
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.mistral_embeddings import MistralEmbedder
from config.settings import (
    DATA_PROCESSED_PATH,
    DATA_EMBEDDINGS_PATH,
    MISTRAL_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration du logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

FICHIER_CSV = Path(DATA_PROCESSED_PATH) / "evenements_nettoyes.csv"
DOSSIER_EMBEDDINGS = Path(DATA_EMBEDDINGS_PATH)
DOSSIER_VERSIONS = DOSSIER_EMBEDDINGS / "versions"
FICHIER_EMBEDDINGS = DOSSIER_EMBEDDINGS / "embeddings.npy"
FICHIER_METADATA = DOSSIER_EMBEDDINGS / "metadata.json"
FICHIER_STATS = DOSSIER_EMBEDDINGS / "stats_embeddings.json"
CHAMP_ID = "id"


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions
# ─────────────────────────────────────────────────────────────────────────────

def charger_evenements() -> pd.DataFrame:
    """
    Charge les événements nettoyés depuis le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les événements nettoyés.

    Raises:
        FileNotFoundError: Si le fichier CSV est introuvable.
        ValueError: Si la colonne 'texte_complet' est absente.
    """
    if not FICHIER_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : '{FICHIER_CSV}'. "
            f"Exécutez d'abord : python project/scripts/nettoyage_data.py"
        )

    df = pd.read_csv(FICHIER_CSV, encoding="utf-8")
    taille_ko = FICHIER_CSV.stat().st_size / 1024

    print(f"\n✓ {len(df)} événements chargés")
    print(f"  Fichier : {FICHIER_CSV}")
    print(f"  Taille  : {taille_ko:.1f} Ko")

    if "texte_complet" not in df.columns:
        raise ValueError(
            "La colonne 'texte_complet' est absente du CSV. "
            "Vérifiez le script de nettoyage."
        )

    if CHAMP_ID not in df.columns:
        raise ValueError(
            f"La colonne '{CHAMP_ID}' est absente du CSV. "
            f"Impossible de dédupliquer les embeddings."
        )

    print(f"\n📋 Colonnes disponibles :")
    for col in df.columns:
        print(f"   • {col}")

    return df


def charger_embeddings_existants() -> Tuple[Optional[np.ndarray], List[Dict]]:
    """
    Charge les embeddings et métadonnées existants si disponibles.

    Returns:
        Tuple[Optional[np.ndarray], List[Dict]]: Les embeddings existants
            et leurs métadonnées, ou (None, []) si absents.
    """
    if not FICHIER_EMBEDDINGS.exists() or not FICHIER_METADATA.exists():
        logger.info("Aucun embedding existant — génération complète.")
        return None, []

    embeddings = np.load(FICHIER_EMBEDDINGS)
    with open(FICHIER_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        "%d embedding(s) existant(s) chargés (dimension=%d).",
        len(embeddings), embeddings.shape[1],
    )
    return embeddings, metadata


def identifier_nouveaux_evenements(
    df: pd.DataFrame,
    metadata_existants: List[Dict],
) -> pd.DataFrame:
    """
    Identifie les événements du DataFrame sans embedding existant.

    Compare les identifiants du DataFrame avec ceux des métadonnées
    existantes pour ne retourner que les événements nouveaux.

    Args:
        df (pd.DataFrame): DataFrame complet des événements nettoyés.
        metadata_existants (List[Dict]): Métadonnées des embeddings
            déjà générés.

    Returns:
        pd.DataFrame: Sous-ensemble du DataFrame contenant uniquement
            les événements sans embedding.
    """
    if not metadata_existants:
        logger.info("Aucun embedding existant — tous les événements sont nouveaux.")
        return df

    ids_existants = {
        str(meta.get(CHAMP_ID))
        for meta in metadata_existants
        if meta.get(CHAMP_ID) is not None
    }

    df_nouveaux = df[~df[CHAMP_ID].astype(str).isin(ids_existants)].copy()

    logger.info(
        "%d événement(s) total, %d déjà traités, %d nouveaux à embedder.",
        len(df), len(ids_existants), len(df_nouveaux),
    )

    return df_nouveaux


def generer_embeddings(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Génère les embeddings pour les événements fournis.

    Args:
        df (pd.DataFrame): DataFrame des événements à embedder.
            Doit contenir les colonnes 'texte_complet' et 'id'.

    Returns:
        Tuple[np.ndarray, List[Dict], Dict]: Un tuple contenant :
            - Les embeddings générés (shape: N x dimension).
            - Les métadonnées associées à chaque embedding.
            - Les statistiques de génération.

    Raises:
        RuntimeError: Si la génération échoue pour tous les batches.
    """
    embedder = MistralEmbedder()
    textes = df["texte_complet"].fillna("").tolist()

    debut = time.time()
    embeddings_array, stats = embedder.generer_embeddings_dataframe(
        df=df,
        colonne_texte="texte_complet",
    )
    duree = time.time() - debut

    stats["duree_secondes"] = round(duree, 2)
    stats["temps_moyen_par_event"] = round(duree / max(len(df), 1), 3)

    # Construire les métadonnées
    colonnes_meta = [
        c for c in ["id", "titre", "ville", "region",
                     "date_debut", "date_fin", "categories", "url"]
        if c in df.columns
    ]
    metadata = df[colonnes_meta].to_dict(orient="records")

    return embeddings_array, metadata, stats


def fusionner_et_sauvegarder(
    emb_existants: Optional[np.ndarray],
    meta_existants: List[Dict],
    nouveaux_emb: np.ndarray,
    nouvelles_meta: List[Dict],
    stats: Dict,
) -> None:
    """
    Fusionne les nouveaux embeddings avec les existants et sauvegarde.

    Crée également un snapshot horodaté dans le dossier `versions/`.

    Args:
        emb_existants (Optional[np.ndarray]): Embeddings existants,
            ou None si première génération.
        meta_existants (List[Dict]): Métadonnées existantes.
        nouveaux_emb (np.ndarray): Nouveaux embeddings générés.
        nouvelles_meta (List[Dict]): Nouvelles métadonnées.
        stats (Dict): Statistiques de génération à sauvegarder.

    Returns:
        None
    """
    DOSSIER_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    DOSSIER_VERSIONS.mkdir(parents=True, exist_ok=True)

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Fusion ────────────────────────────────────────────────────────
    if emb_existants is not None and len(emb_existants) > 0:
        emb_fusionnes = np.vstack([emb_existants, nouveaux_emb])
        meta_fusionnes = meta_existants + nouvelles_meta
        logger.info(
            "Fusion : %d existants + %d nouveaux = %d total.",
            len(emb_existants), len(nouveaux_emb), len(emb_fusionnes),
        )
    else:
        emb_fusionnes = nouveaux_emb
        meta_fusionnes = nouvelles_meta
        logger.info(
            "Première génération : %d embeddings.", len(emb_fusionnes)
        )

    # ── Sauvegarde des fichiers courants ──────────────────────────────
    np.save(FICHIER_EMBEDDINGS, emb_fusionnes)
    logger.info(
        "Embeddings sauvegardés : %s (%.2f Mo)",
        FICHIER_EMBEDDINGS,
        FICHIER_EMBEDDINGS.stat().st_size / (1024 * 1024),
    )

    with open(FICHIER_METADATA, "w", encoding="utf-8") as f:
        json.dump(meta_fusionnes, f, ensure_ascii=False, indent=2)
    logger.info(
        "Métadonnées sauvegardées : %s (%.2f Ko)",
        FICHIER_METADATA,
        FICHIER_METADATA.stat().st_size / 1024,
    )

    # Enrichir les stats avec les totaux fusionnés
    stats["nb_total_embeddings"] = len(emb_fusionnes)
    stats["nb_nouveaux_generes"] = len(nouveaux_emb)
    stats["nb_existants_conserves"] = len(emb_existants) if emb_existants is not None else 0
    stats["horodatage"] = horodatage

    with open(FICHIER_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Statistiques sauvegardées : %s", FICHIER_STATS)

    # ── Snapshot versionné ────────────────────────────────────────────
    fichier_emb_version = DOSSIER_VERSIONS / f"embeddings_{horodatage}.npy"
    fichier_meta_version = DOSSIER_VERSIONS / f"metadata_{horodatage}.json"

    np.save(fichier_emb_version, emb_fusionnes)
    with open(fichier_meta_version, "w", encoding="utf-8") as f:
        json.dump(meta_fusionnes, f, ensure_ascii=False, indent=2)

    logger.info(
        "Snapshot versionné : %s (%d embeddings).",
        fichier_emb_version, len(emb_fusionnes),
    )


def afficher_resume(
    df_total: pd.DataFrame,
    df_nouveaux: pd.DataFrame,
    stats: Dict,
    emb_existants: Optional[np.ndarray],
) -> None:
    """
    Affiche le résumé de la génération dans la console.

    Args:
        df_total (pd.DataFrame): DataFrame complet des événements.
        df_nouveaux (pd.DataFrame): DataFrame des nouveaux événements
            traités lors de cette session.
        stats (Dict): Statistiques de génération.
        emb_existants (Optional[np.ndarray]): Embeddings existants
            avant cette session.

    Returns:
        None
    """
    nb_existants = len(emb_existants) if emb_existants is not None else 0
    nb_nouveaux = len(df_nouveaux)
    nb_total = nb_existants + nb_nouveaux

    print("\n📊 Statistiques :")
    print(f"   Événements dans le CSV     : {len(df_total)}")
    print(f"   Embeddings déjà existants  : {nb_existants}")
    print(f"   Nouveaux embeddings générés: {nb_nouveaux}")
    print(f"   Total embeddings           : {nb_total}")
    print(f"   Dimension                  : {stats.get('dimension', EMBEDDING_DIMENSION)}")
    print(f"   Taille des batches         : {EMBEDDING_BATCH_SIZE}")

    if nb_nouveaux > 0:
        print(f"   Temps total                : {stats.get('duree_secondes', '?')}s")
        print(f"   Temps moyen/événement      : {stats.get('temps_moyen_par_event', '?')}s")
        tokens = stats.get("tokens_utilises", stats.get("total_tokens", "?"))
        print(f"   Tokens utilisés            : {tokens:,}" if isinstance(tokens, int) else f"   Tokens utilisés            : {tokens}")

    erreurs = stats.get("erreurs", [])
    if erreurs:
        print(f"\n⚠️  {len(erreurs)} erreur(s) :")
        for err in erreurs[:5]:
            print(f"   - {err}")
    else:
        print("\n✅ Aucune erreur")

    if nb_nouveaux > 0 and not df_nouveaux.empty:
        premier = df_nouveaux.iloc[0]
        print(f"\n🔍 Exemple d'embedding (premier nouvel événement) :")
        print(f"   Titre : {premier.get('titre', '?')}")
        print(f"   Ville : {premier.get('ville', '?')}")


def main() -> None:
    """
    Fonction principale — génération incrémentale des embeddings.

    Charge les événements nettoyés, identifie ceux sans embedding,
    génère uniquement les embeddings manquants, fusionne avec les
    existants et sauvegarde le tout avec un snapshot versionné.

    Returns:
        None
    """
    print("\n" + "🤖 CRÉATION DES EMBEDDINGS AVEC MISTRAL AI ".center(70, "="))

    try:
        # ── 1. Chargement ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("CHARGEMENT DES DONNÉES")
        print("=" * 70)
        df_total = charger_evenements()

        # ── 2. Embeddings existants ───────────────────────────────────
        print("\n" + "=" * 70)
        print("VÉRIFICATION DES EMBEDDINGS EXISTANTS")
        print("=" * 70)
        emb_existants, meta_existants = charger_embeddings_existants()

        nb_existants = len(emb_existants) if emb_existants is not None else 0
        print(f"\n  Embeddings existants : {nb_existants}")

        # ── 3. Identification des nouveaux événements ─────────────────
        df_nouveaux = identifier_nouveaux_evenements(df_total, meta_existants)
        print(f"  Nouveaux à générer   : {len(df_nouveaux)}")

        if df_nouveaux.empty:
            print("\n✅ Tous les événements ont déjà un embedding.")
            print("   Aucune génération nécessaire.")
            print("\nProchaine étape :")
            print("  python project/scripts/creer_index_faiss.py")
            return

        # ── 4. Génération ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("CRÉATION DES EMBEDDINGS")
        print("=" * 70)
        nouveaux_emb, nouvelles_meta, stats = generer_embeddings(df_nouveaux)

        # ── 5. Fusion et sauvegarde ───────────────────────────────────
        print("\n" + "=" * 70)
        print("SAUVEGARDE DES RÉSULTATS")
        print("=" * 70)
        fusionner_et_sauvegarder(
            emb_existants, meta_existants,
            nouveaux_emb, nouvelles_meta,
            stats,
        )

        # ── 6. Résumé ─────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("RÉSUMÉ")
        print("=" * 70)
        afficher_resume(df_total, df_nouveaux, stats, emb_existants)

    except (FileNotFoundError, ValueError) as e:
        logger.error("Erreur : %s", e)
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ EMBEDDINGS CRÉÉS AVEC SUCCÈS")
    print("=" * 70)
    print("\nFichiers générés :")
    print(f"  • {FICHIER_EMBEDDINGS}")
    print(f"  • {FICHIER_METADATA}")
    print(f"  • {FICHIER_STATS}")
    print(f"  • {DOSSIER_VERSIONS}/embeddings_YYYYMMDD_HHMMSS.npy")
    print("\nProchaine étape :")
    print("  python project/scripts/creer_index_faiss.py")


if __name__ == "__main__":
    main()
