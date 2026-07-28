"""
Script de création de l'index FAISS.

Ce script orchestre le pipeline de création de l'index vectoriel FAISS
à partir des embeddings pré-calculés. Il effectue les étapes suivantes :
    1. Chargement des embeddings et métadonnées.
    2. Création de l'index FAISS.
    3. Test de l'index avec une requête exemple.
    4. Sauvegarde de l'index sur disque.
    5. Affichage d'un résumé des statistiques.

Usage:
    python scripts/creer_index_faiss.py

Prérequis:
    Les embeddings doivent avoir été générés au préalable via :
    python scripts/creer_embeddings.py
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.faiss_index import FAISSIndex
from config.settings import DATA_EMBEDDINGS_PATH, DATA_VECTORSTORE_PATH, FAISS_INDEX_TYPE


def charger_embeddings():
    """
    Charge les embeddings et les métadonnées depuis le dossier de données.

    Lit le fichier 'embeddings.npy' et 'metadata.json' depuis le répertoire
    défini par DATA_EMBEDDINGS_PATH.

    Returns:
        Tuple[np.ndarray, List[Dict]] | Tuple[None, None]:
            - embeddings : Array numpy de forme (N, dimension).
            - metadata   : Liste de dictionnaires de métadonnées.
            Retourne (None, None) si les fichiers sont introuvables.
    """
    print("\n" + "="*70)
    print("CHARGEMENT DES EMBEDDINGS")
    print("="*70)

    embeddings_file = Path(DATA_EMBEDDINGS_PATH) / 'embeddings.npy'
    metadata_file = Path(DATA_EMBEDDINGS_PATH) / 'metadata.json'

    if not embeddings_file.exists():
        print(f"\n❌ Fichier non trouvé : {embeddings_file}")
        print("\nExécutez d'abord :")
        print("  python scripts/creer_embeddings.py")
        return None, None

    embeddings = np.load(embeddings_file)

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"\n✓ Embeddings chargés")
    print(f"  Fichier    : {embeddings_file}")
    print(f"  Nombre     : {len(embeddings)}")
    print(f"  Dimension  : {embeddings.shape[1]}")
    print(f"  Taille     : {embeddings_file.stat().st_size / 1024 / 1024:.2f} Mo")

    print(f"\n✓ Métadonnées chargées")
    print(f"  Fichier    : {metadata_file}")
    print(f"  Nombre     : {len(metadata)}")
    print(f"  Taille     : {metadata_file.stat().st_size / 1024:.2f} Ko")

    return embeddings, metadata


def creer_index(embeddings: np.ndarray) -> FAISSIndex:
    """
    Initialise et crée l'index FAISS à partir des embeddings.

    Args:
        embeddings (np.ndarray): Array numpy des embeddings de forme
            (N, dimension).

    Returns:
        FAISSIndex: Instance de FAISSIndex avec l'index créé et peuplé.
    """
    print("\n" + "="*70)
    print("CRÉATION DE L'INDEX FAISS")
    print("="*70)

    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    faiss_index.creer_index(embeddings, index_type=FAISS_INDEX_TYPE)

    return faiss_index


def tester_index(
    faiss_index: FAISSIndex,
    embeddings: np.ndarray,
    metadata: list
) -> list:
    """
    Teste l'index FAISS avec le premier vecteur comme requête exemple.

    Effectue une recherche des 5 plus proches voisins du premier embedding
    et affiche les résultats avec leurs métadonnées.

    Args:
        faiss_index (FAISSIndex): Instance de FAISSIndex initialisée.
        embeddings (np.ndarray): Array numpy des embeddings.
        metadata (list): Liste des métadonnées associées aux embeddings.

    Returns:
        List[Dict]: Liste des résultats de recherche, chacun contenant
            le rang, la distance, le score et les métadonnées associées.
    """
    print("\n" + "="*70)
    print("TEST DE L'INDEX")
    print("="*70)

    query_embedding = embeddings[0]

    print(f"\n🔍 Requête de test :")
    print(f"   Événement : {metadata[0]['titre']}")
    print(f"   Ville     : {metadata[0]['ville']}")

    resultats = faiss_index.rechercher_avec_metadata(
        query_embedding,
        metadata,
        k=5
    )

    print(f"\n📊 Top 5 résultats :")
    for res in resultats:
        print(f"\n   {res['rank']}. {res['titre']}")
        print(f"      Ville    : {res['ville']}")
        print(f"      Distance : {res['distance']:.4f}")
        print(f"      Score    : {res['score']:.4f}")

    return resultats


def sauvegarder_index(faiss_index: FAISSIndex, metadata: list) -> None:
    """
    Sauvegarde l'index FAISS et les métadonnées sur disque.

    Args:
        faiss_index (FAISSIndex): Instance de FAISSIndex à sauvegarder.
        metadata (list): Métadonnées à associer à l'index sauvegardé.
    """
    print("\n" + "="*70)
    print("SAUVEGARDE DE L'INDEX")
    print("="*70)

    faiss_index.sauvegarder(DATA_VECTORSTORE_PATH, metadata)


def afficher_resume(faiss_index: FAISSIndex) -> None:
    """
    Affiche un résumé des statistiques de l'index FAISS créé.

    Args:
        faiss_index (FAISSIndex): Instance de FAISSIndex dont on souhaite
            afficher les statistiques.
    """
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)

    stats = faiss_index.obtenir_statistiques()

    print(f"\n📊 Statistiques de l'index :")
    print(f"   Type              : {stats['type_index']}")
    print(f"   Dimension         : {stats['dimension']}")
    print(f"   Nombre de vecteurs: {stats['nombre_vecteurs']}")
    print(f"   Entraîné          : {stats.get('is_trained', 'N/A')}")
    print(f"   Nb métadonnées    : {stats.get('nb_metadata', 'N/A')}")
    print(f"   Cohérent          : {stats.get('coherent', 'N/A')}")


def main() -> None:
    """
    Fonction principale du script de création de l'index FAISS.

    Orchestre les étapes suivantes :
        1. Chargement des embeddings et métadonnées.
        2. Création de l'index FAISS.
        3. Test de l'index avec une requête exemple.
        4. Sauvegarde de l'index sur disque.
        5. Affichage du résumé des statistiques.

    Returns:
        None
    """
    print("\n" + "🗂️ CRÉATION DE L'INDEX FAISS ".center(70, "="))

    embeddings, metadata = charger_embeddings()
    if embeddings is None:
        return

    faiss_index = creer_index(embeddings)
    tester_index(faiss_index, embeddings, metadata)
    sauvegarder_index(faiss_index, metadata)
    afficher_resume(faiss_index)

    print("\n" + "="*70)
    print("✅ INDEX FAISS CRÉÉ AVEC SUCCÈS")
    print("="*70)
    print("\nFichiers générés :")
    print(f"  • {DATA_VECTORSTORE_PATH}/faiss.index")
    print(f"  • {DATA_VECTORSTORE_PATH}/metadata.json")
    print(f"  • {DATA_VECTORSTORE_PATH}/stats_index.json")
    print("\nProchaine étape :")
    print("  python scripts/05_query_rag.py")


if __name__ == "__main__":
    main()
