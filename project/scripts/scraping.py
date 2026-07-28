"""
Script de collecte des données depuis l'API OpenAgenda.

Ce script constitue la première étape du pipeline RAG. Il orchestre
la collecte des événements culturels via l'API OpenAgenda et sauvegarde
les données brutes au format JSON pour traitement ultérieur.

Les étapes réalisées sont :
    1. Lecture du curseur de pagination de la session précédente.
    2. Connexion à l'API OpenAgenda via OpenAgendaClient.
    3. Récupération de la page d'agendas suivante (pagination par curseur).
    4. Collecte des événements pour chaque agenda.
    5. Fusion et déduplication avec les données existantes.
    6. Sauvegarde du curseur pour la prochaine session.

Usage:
    python scripts/scraping.py

Prérequis:
    - La variable d'environnement OPENAGENDA_API_KEY doit être définie
      dans le fichier .env à la racine du projet.

Prochaine étape:
    python scripts/nettoyage_data.py
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.openagenda_client import OpenAgendaClient
from src.data_collection.data_versioning import DataVersioning
from config.settings import (
    OPENAGENDA_NB_AGENDAS,
    OPENAGENDA_EVENTS_PAR_AGENDA,
    DATA_RAW_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration du logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions
# ─────────────────────────────────────────────────────────────────────────────

def collecter_evenements(
    after: Optional[list] = None,
) -> Tuple[List[dict], Optional[list], int]:
    """
    Collecte les événements depuis l'API OpenAgenda avec pagination
    par curseur.

    Récupère une page d'agendas à partir du curseur `after`, puis
    collecte les événements de chaque agenda trouvé.

    Args:
        after (Optional[list], optionnel): Curseur de pagination
            [index, uid] retourné par l'appel précédent. None pour
            démarrer depuis le début de la liste des agendas.

    Returns:
        Tuple[List[dict], Optional[list], int]: Un tuple contenant :
            - La liste de tous les événements collectés.
            - Le curseur `after` pour la prochaine page (None si fin
              de liste atteinte).
            - Le nombre total d'agendas disponibles selon l'API.

    Raises:
        RuntimeError: Si le client OpenAgenda ne peut pas être instancié.
    """
    logger.info("Initialisation du client OpenAgenda...")

    try:
        client = OpenAgendaClient()
    except ValueError as e:
        raise RuntimeError(
            f"Impossible d'initialiser le client OpenAgenda : {e}"
        ) from e

    # Utilisation de %s pour supporter None et list sans TypeError
    logger.info(
        "Récupération des agendas (limite=%d, after=%s)...",
        OPENAGENDA_NB_AGENDAS, after,
    )

    agendas, data_brute = client.get_agendas(
        limite=OPENAGENDA_NB_AGENDAS,
        after=after,
    )

    if not agendas:
        logger.warning("Aucun agenda récupéré (after=%s).", after)
        return [], None, 0

    curseur_suivant = data_brute.get("after")
    total_api = data_brute.get("total", 0)

    logger.info(
        "%d agenda(s) récupéré(s) — prochain curseur : %s — total : %d.",
        len(agendas), curseur_suivant, total_api,
    )
    print(
        f"\n✓ {len(agendas)} agendas récupérés "
        f"(curseur={after}, total API={total_api})\n"
    )

    # ── Collecte des événements ───────────────────────────────────────
    all_events: List[dict] = []
    agendas_avec_events: int = 0
    agendas_en_erreur: int = 0

    for i, agenda in enumerate(agendas, 1):
        agenda_uid = agenda.get("uid")
        agenda_title = agenda.get("title", "Sans titre")

        print(f"{i:2}. {agenda_title[:50]}")

        if not agenda_uid:
            logger.warning("Agenda sans UID ignoré : %s", agenda_title)
            print(f"    ⚠ UID manquant — agenda ignoré")
            agendas_en_erreur += 1
            continue

        events = client.get_events_from_agenda(
            agenda_uid,
            limite=OPENAGENDA_EVENTS_PAR_AGENDA,
        )

        if events:
            print(f"    ✓ {len(events)} événement(s)")
            logger.info(
                "Agenda '%s' (uid=%s) : %d événement(s).",
                agenda_title, agenda_uid, len(events),
            )
            all_events.extend(events)
            agendas_avec_events += 1
        else:
            print(f"    ⚠ Aucun événement")
            logger.warning(
                "Agenda '%s' (uid=%s) : aucun événement.",
                agenda_title, agenda_uid,
            )

    # ── Résumé ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RÉSUMÉ DE LA COLLECTE")
    print("=" * 70)
    print(f"  Curseur utilisé         : {after}")
    print(f"  Prochain curseur        : {curseur_suivant}")
    print(f"  Total agendas (API)     : {total_api}")
    print(f"  Agendas explorés        : {len(agendas)}")
    print(f"  Agendas avec événements : {agendas_avec_events}")
    print(f"  Agendas sans UID        : {agendas_en_erreur}")
    print(f"  Total événements        : {len(all_events)}")

    logger.info(
        "Collecte terminée — %d événement(s), prochain curseur=%s.",
        len(all_events), curseur_suivant,
    )

    return all_events, curseur_suivant, total_api


def sauvegarder_donnees_brutes(events: List[dict]) -> None:
    """
    Sauvegarde les événements collectés au format JSON sur disque.

    Args:
        events (List[dict]): Liste des événements à sauvegarder.
            Si vide, la sauvegarde est ignorée.

    Returns:
        None

    Raises:
        OSError: Si l'écriture du fichier échoue.
    """
    if not events:
        logger.warning("Liste vide — aucune sauvegarde effectuée.")
        print("\n⚠ Aucun événement à sauvegarder.")
        return

    output_path = Path(DATA_RAW_PATH)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Impossible de créer '%s' : %s", output_path, e)
        raise

    output_file = output_path / "evenements_bruts.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error("Échec de l'écriture dans '%s' : %s", output_file, e)
        raise

    taille_ko = output_file.stat().st_size / 1024
    logger.info(
        "Données sauvegardées : '%s' (%.1f Ko, %d événements).",
        output_file, taille_ko, len(events),
    )
    print(f"\n✓ Données sauvegardées : {output_file}")
    print(f"  Nombre d'événements : {len(events)}")
    print(f"  Taille              : {taille_ko:.1f} Ko")


def _charger_events_fusionnes() -> List[dict]:
    """
    Charge le fichier courant des événements fusionnés.

    Returns:
        List[dict]: Liste complète des événements fusionnés,
            ou liste vide si le fichier est absent.
    """
    fichier = Path(DATA_RAW_PATH) / "evenements_bruts.json"
    if not fichier.exists():
        return []
    with open(fichier, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """
    Fonction principale du script de collecte OpenAgenda avec pagination
    par curseur.

    Orchestre les étapes suivantes :
        1. Lecture du curseur de la session précédente.
        2. Collecte des événements (page suivante d'agendas).
        3. Fusion et déduplication avec les données existantes.
        4. Sauvegarde du curseur pour la prochaine session.
        5. Sauvegarde du fichier brut courant (cumul fusionné).

    Returns:
        None
    """
    print("\n" + "COLLECTE DE DONNÉES OPEN AGENDA ".center(70, "="))
    print()

    try:
        versioner = DataVersioning()

        # 1. Lire le curseur de la session précédente
        after = versioner.lire_offset()

        # ✅ Correction : %s au lieu de %d pour supporter None et list
        logger.info("Reprise de la collecte à partir du curseur : %s.", after)

        # 2. Collecter — 3 valeurs retournées
        events, curseur_suivant, total_api = collecter_evenements(after=after)

        if events:
            # 3. Fusionner et versionner
            rapport = versioner.integrer_nouvelle_collecte(events)

            # 4. Sauvegarder le curseur suivant
            versioner.sauvegarder_offset(
                curseur_after=curseur_suivant,
                nb_agendas_recus=OPENAGENDA_NB_AGENDAS,
                total_api=total_api,
            )

            # 5. Afficher le rapport
            label_curseur = (
                str(curseur_suivant)
                if curseur_suivant
                else "(fin de liste → reset au prochain appel)"
            )
            print("\n" + "=" * 70)
            print("RAPPORT D'INTÉGRATION")
            print("=" * 70)
            print(f"  Curseur utilisé       : {after}")
            print(f"  Prochain curseur      : {label_curseur}")
            print(f"  Événements collectés  : {rapport['nb_nouveaux']}")
            print(f"  Avant fusion          : {rapport['nb_avant_fusion']}")
            print(f"  Après fusion          : {rapport['nb_apres_fusion']}")
            print(f"  Nouveaux ajoutés      : {rapport['nb_ajoutes']}")
            print(f"  Mis à jour            : {rapport['nb_mis_a_jour']}")
            print(f"  Doublons ignorés      : {rapport['nb_doublons']}")
            print(f"  Champ ID utilisé      : {rapport.get('champ_id_utilise', '?')}")

            # 6. Sauvegarder le fichier brut courant (cumul complet)
            events_fusionnes = _charger_events_fusionnes()
            sauvegarder_donnees_brutes(events_fusionnes)

        else:
            # Aucun résultat → reset du curseur
            versioner.sauvegarder_offset(
                curseur_after=None,
                nb_agendas_recus=0,
                total_api=total_api,
            )
            print("\n⚠ Aucun événement collecté.")
            print("  Le curseur a été remis à None pour la prochaine session.")

    except RuntimeError as e:
        logger.error("Erreur critique : %s", e)
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)
    except OSError as e:
        logger.error("Erreur d'écriture : %s", e)
        print(f"\n❌ Erreur de sauvegarde : {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ COLLECTE ET VERSIONNEMENT TERMINÉS")
    print("=" * 70)
    print("\nProchaine étape :")
    print("  python scripts/nettoyage_data.py")


if __name__ == "__main__":
    main()