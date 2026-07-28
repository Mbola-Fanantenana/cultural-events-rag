"""
Module de versionnement et fusion des données collectées.

Ce module fournit la classe :class:`DataVersionner`, responsable de :
    - Sauvegarder chaque collecte dans un sous-dossier horodaté.
    - Fusionner les nouvelles données avec les données existantes.
    - Dédupliquer les événements par identifiant unique.
    - Maintenir un journal des collectes (collecte_log.json).
    - Reconstruire l'index FAISS de manière incrémentale.

Il s'insère entre le script de collecte (`scripts/scraping.py`) et
le pipeline d'embedding/indexation.

Usage typique:
    >>> versioner = DataVersionner()
    >>> rapport = versioner.integrer_nouvelle_collecte(nouveaux_events)
    >>> print(rapport)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    DATA_RAW_PATH,
    DATA_PROCESSED_PATH,
    DATA_EMBEDDINGS_PATH,
    DATA_VECTORSTORE_PATH,
)

logger = logging.getLogger(__name__)

# Nom du champ utilisé comme identifiant unique des événements
CHAMP_ID = "uid"

# Nom du sous-dossier de versionnement dans chaque répertoire de données
DOSSIER_VERSIONS = "versions"

# Nom du fichier journal des collectes
FICHIER_LOG = "collecte_log.json"


class DataVersioning:
    """
    Gère le versionnement et la fusion des données collectées.

    Chaque appel à :meth:`integrer_nouvelle_collecte` :
        1. Sauvegarde les nouveaux événements dans un fichier horodaté.
        2. Charge les données existantes (si présentes).
        3. Fusionne et déduplique par `uid`.
        4. Sauvegarde le fichier fusionné comme fichier courant.
        5. Met à jour le journal des collectes.

    Attributes:
        raw_path (Path): Dossier des données brutes.
        processed_path (Path): Dossier des données nettoyées.
        embeddings_path (Path): Dossier des embeddings.
        vectorstore_path (Path): Dossier de l'index vectoriel.
        horodatage (str): Horodatage de la session courante (YYYYMMDD_HHMMSS).
    """

    def __init__(self) -> None:
        """
        Initialise le DataVersionner et crée les dossiers nécessaires.
        """
        self.raw_path = Path(DATA_RAW_PATH)
        self.processed_path = Path(DATA_PROCESSED_PATH)
        self.embeddings_path = Path(DATA_EMBEDDINGS_PATH)
        self.vectorstore_path = Path(DATA_VECTORSTORE_PATH)
        self.horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._creer_dossiers()
        logger.info("DataVersionner initialisé (session : %s).", self.horodatage)

    # ─────────────────────────────────────────────────────────────────────
    # API publique
    # ─────────────────────────────────────────────────────────────────────

    def integrer_nouvelle_collecte(
        self, nouveaux_events: List[Dict]
    ) -> Dict:
        """
        Intègre une nouvelle collecte dans les données existantes.

        Sauvegarde les nouveaux événements dans un fichier versionné,
        fusionne avec les données existantes en dédupliquant par `uid`,
        puis met à jour le fichier courant et le journal.

        Args:
            nouveaux_events (List[Dict]): Liste des événements issus de
                la dernière collecte OpenAgenda.

        Returns:
            Dict: Rapport d'intégration contenant :
                - horodatage (str)       : Horodatage de la session.
                - nb_nouveaux (int)      : Nombre d'événements collectés.
                - nb_avant_fusion (int)  : Nombre d'événements avant fusion.
                - nb_apres_fusion (int)  : Nombre d'événements après fusion.
                - nb_ajoutes (int)       : Nombre de nouveaux événements ajoutés.
                - nb_mis_a_jour (int)    : Nombre d'événements mis à jour.
                - nb_doublons (int)      : Nombre de doublons ignorés.
                - fichier_version (str)  : Chemin du fichier versionné.

        Raises:
            ValueError: Si `nouveaux_events` est vide.
            OSError: Si l'écriture des fichiers échoue.
        """
        if not nouveaux_events:
            raise ValueError(
                "La liste de nouveaux événements est vide. "
                "Aucune intégration effectuée."
            )

        logger.info(
            "Intégration de %d nouveaux événements (session %s)...",
            len(nouveaux_events), self.horodatage,
        )

        # 1. Sauvegarder la version brute horodatée
        fichier_version = self._sauvegarder_version_brute(nouveaux_events)

        # 2. Charger les données existantes
        events_existants = self._charger_events_existants()
        nb_avant = len(events_existants)

        # 3. Fusionner et dédupliquer
        events_fusionnes, rapport_fusion = self._fusionner_events(
            events_existants, nouveaux_events
        )
        nb_apres = len(events_fusionnes)

        # 4. Sauvegarder le fichier courant fusionné
        self._sauvegarder_events_courants(events_fusionnes)

        # 5. Mettre à jour le journal
        rapport = {
            "horodatage": self.horodatage,
            "nb_nouveaux": len(nouveaux_events),
            "nb_avant_fusion": nb_avant,
            "nb_apres_fusion": nb_apres,
            "nb_ajoutes": rapport_fusion["ajoutes"],
            "nb_mis_a_jour": rapport_fusion["mis_a_jour"],
            "nb_doublons": rapport_fusion["doublons"],
            "fichier_version": str(fichier_version),
        }
        self._mettre_a_jour_journal(rapport)

        logger.info(
            "Intégration terminée : %d → %d événements "
            "(%d ajoutés, %d mis à jour, %d doublons ignorés).",
            nb_avant, nb_apres,
            rapport_fusion["ajoutes"],
            rapport_fusion["mis_a_jour"],
            rapport_fusion["doublons"],
        )

        return rapport

    def integrer_embeddings_et_index(
        self,
        nouveaux_embeddings: np.ndarray,
        nouvelles_metadata: List[Dict],
        uids_nouveaux: List[str],
    ) -> None:
        """
        Intègre de nouveaux embeddings dans les fichiers existants et
        reconstruit l'index FAISS de manière incrémentale.

        Sauvegarde une version horodatée des embeddings, fusionne avec
        les embeddings existants (en dédupliquant par uid), puis
        reconstruit l'index FAISS complet.

        Args:
            nouveaux_embeddings (np.ndarray): Array numpy de forme
                (N, dimension) contenant les nouveaux vecteurs.
            nouvelles_metadata (List[Dict]): Métadonnées associées aux
                nouveaux embeddings (même ordre, même longueur).
            uids_nouveaux (List[str]): Liste des UIDs correspondant aux
                nouveaux embeddings, utilisée pour la déduplication.

        Returns:
            None

        Raises:
            ValueError: Si les longueurs de nouveaux_embeddings,
                nouvelles_metadata et uids_nouveaux ne correspondent pas.
            OSError: Si l'écriture des fichiers échoue.
        """
        if not (
            len(nouveaux_embeddings)
            == len(nouvelles_metadata)
            == len(uids_nouveaux)
        ):
            raise ValueError(
                "Les longueurs de nouveaux_embeddings, nouvelles_metadata "
                "et uids_nouveaux doivent être identiques."
            )

        logger.info(
            "Intégration de %d nouveaux embeddings...", len(nouveaux_embeddings)
        )

        # 1. Sauvegarder la version horodatée des embeddings
        self._sauvegarder_version_embeddings(
            nouveaux_embeddings, nouvelles_metadata
        )

        # 2. Charger les embeddings existants
        emb_existants, meta_existants = self._charger_embeddings_existants()

        # 3. Fusionner en dédupliquant par uid
        emb_fusionnes, meta_fusionnes = self._fusionner_embeddings(
            emb_existants, meta_existants,
            nouveaux_embeddings, nouvelles_metadata,
            uids_nouveaux,
        )

        # 4. Sauvegarder les embeddings fusionnés
        self._sauvegarder_embeddings_courants(emb_fusionnes, meta_fusionnes)

        # 5. Reconstruire l'index FAISS
        self._reconstruire_index_faiss(emb_fusionnes, meta_fusionnes)

        logger.info(
            "Embeddings intégrés : %d → %d vecteurs.",
            len(emb_existants) if emb_existants is not None else 0,
            len(emb_fusionnes),
        )

    def lister_versions(self, type_donnee: str = "raw") -> List[Dict]:
        """
        Liste les versions disponibles pour un type de données.

        Args:
            type_donnee (str): Type de données à lister. Valeurs possibles :
                - "raw"        : Données brutes.
                - "processed"  : Données nettoyées.
                - "embeddings" : Embeddings.
                - "vectorstore": Index vectoriel.

        Returns:
            List[Dict]: Liste de dictionnaires triés par date décroissante,
                chacun contenant :
                - fichier (str)      : Nom du fichier.
                - chemin (str)       : Chemin complet.
                - date (str)         : Date extraite du nom de fichier.
                - taille_ko (float)  : Taille en Ko.

        Raises:
            ValueError: Si `type_donnee` n'est pas une valeur reconnue.
        """
        chemins = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "embeddings": self.embeddings_path,
            "vectorstore": self.vectorstore_path,
        }

        if type_donnee not in chemins:
            raise ValueError(
                f"Type de données inconnu : '{type_donnee}'. "
                f"Valeurs possibles : {list(chemins.keys())}"
            )

        dossier_versions = chemins[type_donnee] / DOSSIER_VERSIONS
        if not dossier_versions.exists():
            return []

        versions = []
        for fichier in sorted(dossier_versions.iterdir(), reverse=True):
            if fichier.is_file() and not fichier.name.startswith("."):
                versions.append({
                    "fichier": fichier.name,
                    "chemin": str(fichier),
                    "taille_ko": round(fichier.stat().st_size / 1024, 1),
                })

        return versions

    def afficher_journal(self) -> None:
        """
        Affiche le journal des collectes dans la console.

        Lit le fichier `collecte_log.json` et affiche chaque entrée
        avec ses statistiques principales.

        Returns:
            None
        """
        log_file = self.raw_path / FICHIER_LOG
        if not log_file.exists():
            print("Aucun journal de collecte trouvé.")
            return

        with open(log_file, "r", encoding="utf-8") as f:
            journal = json.load(f)

        print("\n" + "=" * 70)
        print("JOURNAL DES COLLECTES")
        print("=" * 70)
        print(f"{'Session':<20} {'Nouveaux':>9} {'Avant':>7} {'Après':>7} {'Ajoutés':>8} {'MàJ':>5}")
        print("-" * 70)

        for entree in journal:
            print(
                f"{entree['horodatage']:<20} "
                f"{entree['nb_nouveaux']:>9} "
                f"{entree['nb_avant_fusion']:>7} "
                f"{entree['nb_apres_fusion']:>7} "
                f"{entree['nb_ajoutes']:>8} "
                f"{entree['nb_mis_a_jour']:>5}"
            )

        print("-" * 70)
        print(f"Total sessions : {len(journal)}")

    # ─────────────────────────────────────────────────────────────────────
    # Méthodes privées — Données brutes
    # ─────────────────────────────────────────────────────────────────────

    def _creer_dossiers(self) -> None:
        """
        Crée les dossiers de données et de versions s'ils n'existent pas.
        """
        for chemin in [
            self.raw_path,
            self.processed_path,
            self.embeddings_path,
            self.vectorstore_path,
        ]:
            chemin.mkdir(parents=True, exist_ok=True)
            (chemin / DOSSIER_VERSIONS).mkdir(parents=True, exist_ok=True)

    def lire_offset(self) -> Optional[list]:
        """
        Lit le curseur de pagination depuis le fichier de suivi.

        Le curseur est une liste de deux éléments [index, uid] retournée
        par l'API OpenAgenda dans le champ `after` de chaque réponse.
        Il doit être passé tel quel à l'appel suivant pour obtenir la
        page d'agendas suivante.

        Returns:
            Optional[list]: Le curseur [index, uid] de la session
                précédente, ou None si c'est la première collecte
                (démarrage depuis le début de la liste).
        """
        from config.settings import OPENAGENDA_OFFSET_FILE

        offset_file = Path(OPENAGENDA_OFFSET_FILE)

        if not offset_file.exists():
            logger.info("Aucun fichier de curseur — démarrage depuis le début.")
            return None

        with open(offset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        curseur = data.get("curseur_after")
        horodatage = data.get("horodatage", "inconnue")
        total_connu = data.get("total_agendas_api", "?")

        if curseur is None:
            logger.info(
                "Curseur None dans le fichier — démarrage depuis le début "
                "(session précédente : %s).", horodatage,
            )
        else:
            logger.info(
                "Curseur chargé : %s (total API connu : %s, session : %s).",
                curseur, total_connu, horodatage,
            )

        return curseur


    def sauvegarder_offset(
        self,
        curseur_after: Optional[list],
        nb_agendas_recus: int,
        total_api: int = 0,
    ) -> None:
        """
        Sauvegarde le curseur de pagination pour la prochaine session.

        Si le curseur retourné par l'API est None ou absent, cela signifie
        que la fin de la liste des agendas a été atteinte. L'offset est
        alors remis à None pour recommencer depuis le début lors de la
        prochaine collecte.

        Args:
            curseur_after (Optional[list]): Curseur [index, uid] retourné
                par l'API dans le champ `after` de la dernière réponse.
                None si fin de liste atteinte.
            nb_agendas_recus (int): Nombre d'agendas reçus lors de cette
                session (pour information dans le fichier de suivi).
            total_api (int, optionnel): Nombre total d'agendas disponibles
                selon l'API. Défaut : 0 (inconnu).

        Returns:
            None
        """
        from config.settings import OPENAGENDA_OFFSET_FILE

        offset_file = Path(OPENAGENDA_OFFSET_FILE)
        offset_file.parent.mkdir(parents=True, exist_ok=True)

        fin_de_liste = curseur_after is None

        data = {
            "curseur_after": curseur_after,
            "nb_agendas_session": nb_agendas_recus,
            "total_agendas_api": total_api,
            "horodatage": self.horodatage,
            "fin_de_liste": fin_de_liste,
        }

        with open(offset_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if fin_de_liste:
            logger.info(
                "Fin de liste atteinte — curseur remis à None. "
                "Prochaine session reprendra depuis le début."
            )
        else:
            logger.info(
                "Curseur sauvegardé : %s (total API=%d).",
                curseur_after, total_api,
            )



    def _sauvegarder_version_brute(
        self, events: List[Dict]
    ) -> Path:
        """
        Sauvegarde les événements dans un fichier versionné horodaté.

        Args:
            events (List[Dict]): Événements à sauvegarder.

        Returns:
            Path: Chemin du fichier versionné créé.
        """
        fichier = (
            self.raw_path
            / DOSSIER_VERSIONS
            / f"evenements_bruts_{self.horodatage}.json"
        )
        with open(fichier, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        logger.info(
            "Version brute sauvegardée : %s (%.1f Ko).",
            fichier, fichier.stat().st_size / 1024,
        )
        return fichier

    def _charger_events_existants(self) -> List[Dict]:
        """
        Charge les événements du fichier courant s'il existe.

        Returns:
            List[Dict]: Liste des événements existants, ou liste vide
                si le fichier n'existe pas encore.
        """
        fichier = self.raw_path / "evenements_bruts.json"
        if not fichier.exists():
            logger.info("Aucun fichier existant — première collecte.")
            return []

        with open(fichier, "r", encoding="utf-8") as f:
            events = json.load(f)

        logger.info("%d événement(s) existant(s) chargés.", len(events))
        return events

    def _fusionner_events(
        self,
        existants: List[Dict],
        nouveaux: List[Dict],
    ) -> Tuple[List[Dict], Dict]:
        """
        Fusionne deux listes d'événements en dédupliquant par `uid`.

        Les événements existants sont conservés. Les nouveaux événements
        sont ajoutés s'ils sont absents, ou remplacent l'existant s'ils
        ont le même `uid` (mise à jour).

        Args:
            existants (List[Dict]): Événements déjà présents.
            nouveaux (List[Dict]): Nouveaux événements à intégrer.

        Returns:
            Tuple[List[Dict], Dict]: Un tuple contenant :
                - La liste fusionnée et dédupliquée.
                - Un rapport avec les compteurs ajoutes, mis_a_jour,
                  doublons.
        """
        # Indexer les existants par uid pour accès O(1)
        index_existants: Dict[str, int] = {
            evt.get(CHAMP_ID): i
            for i, evt in enumerate(existants)
            if evt.get(CHAMP_ID)
        }

        fusionnes = list(existants)
        ajoutes = 0
        mis_a_jour = 0
        doublons = 0

        for evt in nouveaux:
            uid = evt.get(CHAMP_ID)

            if not uid:
                # Événement sans UID : on l'ajoute sans déduplication
                fusionnes.append(evt)
                ajoutes += 1
                logger.debug("Événement sans UID ajouté directement.")
                continue

            if uid in index_existants:
                # Mise à jour de l'événement existant
                idx = index_existants[uid]
                if fusionnes[idx] != evt:
                    fusionnes[idx] = evt
                    mis_a_jour += 1
                else:
                    doublons += 1
            else:
                # Nouvel événement
                index_existants[uid] = len(fusionnes)
                fusionnes.append(evt)
                ajoutes += 1

        rapport = {
            "ajoutes": ajoutes,
            "mis_a_jour": mis_a_jour,
            "doublons": doublons,
        }
        return fusionnes, rapport

    def _sauvegarder_events_courants(self, events: List[Dict]) -> None:
        """
        Sauvegarde la liste fusionnée comme fichier courant.

        Args:
            events (List[Dict]): Liste fusionnée des événements.
        """
        fichier = self.raw_path / "evenements_bruts.json"
        with open(fichier, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        logger.info(
            "Fichier courant mis à jour : %s (%d événements, %.1f Ko).",
            fichier, len(events), fichier.stat().st_size / 1024,
        )

    def _mettre_a_jour_journal(self, rapport: Dict) -> None:
        """
        Ajoute une entrée au journal des collectes.

        Args:
            rapport (Dict): Rapport de la collecte courante à journaliser.
        """
        log_file = self.raw_path / FICHIER_LOG
        journal = []

        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                journal = json.load(f)

        journal.append(rapport)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(journal, f, ensure_ascii=False, indent=2)

        logger.info("Journal mis à jour (%d entrée(s)).", len(journal))

    # ─────────────────────────────────────────────────────────────────────
    # Méthodes privées — Embeddings et index
    # ─────────────────────────────────────────────────────────────────────

    def _sauvegarder_version_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Sauvegarde une version horodatée des embeddings et métadonnées.

        Args:
            embeddings (np.ndarray): Array numpy des embeddings.
            metadata (List[Dict]): Métadonnées associées.
        """
        dossier = self.embeddings_path / DOSSIER_VERSIONS
        np.save(dossier / f"embeddings_{self.horodatage}.npy", embeddings)

        with open(
            dossier / f"metadata_{self.horodatage}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "Version embeddings sauvegardée : embeddings_%s.npy (%d vecteurs).",
            self.horodatage, len(embeddings),
        )

    def _charger_embeddings_existants(
        self,
    ) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Charge les embeddings et métadonnées du fichier courant.

        Returns:
            Tuple[Optional[np.ndarray], List[Dict]]: Les embeddings
                existants et leurs métadonnées, ou (None, []) si absents.
        """
        emb_file = self.embeddings_path / "embeddings.npy"
        meta_file = self.embeddings_path / "metadata.json"

        if not emb_file.exists() or not meta_file.exists():
            logger.info("Aucun embedding existant — première génération.")
            return None, []

        embeddings = np.load(emb_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.info("%d embedding(s) existant(s) chargés.", len(embeddings))
        return embeddings, metadata

    def _fusionner_embeddings(
        self,
        emb_existants: Optional[np.ndarray],
        meta_existants: List[Dict],
        nouveaux_emb: np.ndarray,
        nouvelles_meta: List[Dict],
        uids_nouveaux: List[str],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Fusionne les embeddings existants avec les nouveaux en dédupliquant
        par uid.

        Args:
            emb_existants (Optional[np.ndarray]): Embeddings existants,
                ou None si première génération.
            meta_existants (List[Dict]): Métadonnées existantes.
            nouveaux_emb (np.ndarray): Nouveaux embeddings à intégrer.
            nouvelles_meta (List[Dict]): Nouvelles métadonnées.
            uids_nouveaux (List[str]): UIDs des nouveaux embeddings.

        Returns:
            Tuple[np.ndarray, List[Dict]]: Embeddings et métadonnées
                fusionnés et dédupliqués.
        """
        if emb_existants is None or len(emb_existants) == 0:
            return nouveaux_emb, nouvelles_meta

        # UIDs existants pour déduplication
        uids_existants = {
            meta.get(CHAMP_ID)
            for meta in meta_existants
            if meta.get(CHAMP_ID)
        }

        # Filtrer les nouveaux embeddings non présents
        indices_a_ajouter = [
            i for i, uid in enumerate(uids_nouveaux)
            if uid not in uids_existants
        ]

        if not indices_a_ajouter:
            logger.info("Aucun nouvel embedding à ajouter (tous déjà présents).")
            return emb_existants, meta_existants

        emb_a_ajouter = nouveaux_emb[indices_a_ajouter]
        meta_a_ajouter = [nouvelles_meta[i] for i in indices_a_ajouter]

        emb_fusionnes = np.vstack([emb_existants, emb_a_ajouter])
        meta_fusionnes = meta_existants + meta_a_ajouter

        logger.info(
            "%d nouvel(s) embedding(s) ajouté(s) (%d ignorés car déjà présents).",
            len(indices_a_ajouter),
            len(nouveaux_emb) - len(indices_a_ajouter),
        )

        return emb_fusionnes, meta_fusionnes

    def _sauvegarder_embeddings_courants(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Sauvegarde les embeddings fusionnés comme fichiers courants.

        Args:
            embeddings (np.ndarray): Embeddings fusionnés.
            metadata (List[Dict]): Métadonnées fusionnées.
        """
        np.save(self.embeddings_path / "embeddings.npy", embeddings)

        with open(
            self.embeddings_path / "metadata.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "Embeddings courants mis à jour : %d vecteurs.", len(embeddings)
        )

    def _reconstruire_index_faiss(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Reconstruit l'index FAISS complet à partir des embeddings fusionnés
        et sauvegarde une version horodatée.

        Args:
            embeddings (np.ndarray): Embeddings complets fusionnés.
            metadata (List[Dict]): Métadonnées associées.
        """
        from src.vectorstore.faiss_index import FAISSIndex
        from config.settings import FAISS_INDEX_TYPE

        dimension = embeddings.shape[1]
        fi = FAISSIndex(dimension=dimension)
        fi.reconstruire_index(embeddings, metadata, index_type=FAISS_INDEX_TYPE)

        # Sauvegarder la version courante
        fi.sauvegarder(str(self.vectorstore_path), metadata)

        # Sauvegarder une version horodatée
        import faiss
        version_file = (
            self.vectorstore_path
            / DOSSIER_VERSIONS
            / f"faiss_{self.horodatage}.index"
        )
        faiss.write_index(fi.index, str(version_file))

        logger.info(
            "Index FAISS reconstruit et versionné : %s (%d vecteurs).",
            version_file, fi.index.ntotal,
        )
