"""
Module de génération d'embeddings avec l'API Mistral AI.

Ce module fournit la classe :class:`MistralEmbedder`, responsable de :
    - transformer des textes (ou une colonne d'un DataFrame) en vecteurs
      d'embedding via l'API Mistral,
    - traiter les textes par batchs, avec limitation de débit et
      nouvelles tentatives automatiques en cas d'erreur transitoire,
    - sauvegarder/charger les embeddings et leurs métadonnées associées.

Ce module constitue la deuxième étape du pipeline RAG, après le nettoyage
des données (voir `src/data_collection/data_cleaner.py`) et avant la
création de l'index vectoriel (voir `src/vectorstore/faiss_index.py`).
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

try:
    # Chemin d'import valable sur les versions récentes du SDK (>= 2.x)
    from mistralai.client import Mistral
except ImportError:
    # Chemin d'import exposé par certaines versions du SDK (raccourci à la racine)
    from mistralai import Mistral

logger = logging.getLogger(__name__)

# Nombre maximal de tentatives par batch en cas d'erreur transitoire
# (rate limiting, coupure réseau, timeout...).
MAX_RETRIES = 3

# Base du backoff exponentiel entre deux tentatives (en secondes) :
# tentative 1 -> 2s, tentative 2 -> 4s, etc.
RETRY_BACKOFF_BASE = 2.0


class MistralEmbedder:
    """
    Génère des embeddings pour les événements avec l'API Mistral AI.

    Attributes:
        api_key (str): Clé API Mistral utilisée pour l'authentification.
        model (str): Nom du modèle d'embedding utilisé.
        client (Mistral): Client HTTP de la SDK Mistral.
        dimension (int | None): Dimension des vecteurs d'embedding,
            déterminée dynamiquement lors du premier appel réussi à
            l'API. `None` tant qu'aucun embedding n'a été généré avec
            succès.
        stats (dict): Statistiques cumulées de génération (nombre
            d'événements, tokens utilisés, temps total, erreurs...).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_proxy: bool = True,
        verify_ssl: bool = True,
    ) -> None:
        """
        Initialise le client Mistral.

        Args:
            api_key (str, optionnel): Clé API Mistral. Si absente, la clé
                est lue depuis `config.settings.MISTRAL_API_KEY` (elle-même
                généralement définie via la variable d'environnement
                `MISTRAL_API_KEY`).
            model (str, optionnel): Nom du modèle d'embedding. Si absent,
                lu depuis `config.settings.MISTRAL_EMBEDDING_MODEL`.
            use_proxy (bool, optionnel): Si True (défaut), configure un
                proxy HTTP/HTTPS à partir de `config.settings.PROXIES`
                lorsqu'il est renseigné.
            verify_ssl (bool, optionnel): Si False, désactive la
                vérification des certificats SSL/TLS pour les requêtes
                vers l'API Mistral. Défaut : True.

        Raises:
            ValueError: Si aucune clé API n'est disponible (ni en
                paramètre, ni dans la configuration).
        """
        from config.settings import MISTRAL_API_KEY, MISTRAL_EMBEDDING_MODEL, PROXIES

        self.api_key = api_key or MISTRAL_API_KEY

        if not self.api_key:
            raise ValueError(
                "Clé API Mistral manquante. "
                "Définissez MISTRAL_API_KEY dans .env"
            )

        self.model = model or MISTRAL_EMBEDDING_MODEL

        # Dimension des embeddings, déterminée dynamiquement au premier
        # succès (voir _embed_batch_avec_retry). On ne la fige pas en dur
        # pour rester robuste à un changement de modèle.
        self.dimension: Optional[int] = None

        # Construction d'un client HTTP explicite (proxy + vérification
        # SSL), passé à la SDK Mistral plutôt que de bidouiller des
        # variables d'environnement globales (qui affectent tout le
        # process, pas seulement les appels à Mistral).
        http_client_kwargs: Dict[str, Any] = {}

        if use_proxy and PROXIES:
            proxy_url = PROXIES.get('https://') or PROXIES.get('http://')
            if proxy_url:
                logger.info("Configuration du proxy : %s", proxy_url)
                http_client_kwargs['proxy'] = proxy_url
            else:
                logger.warning("use_proxy=True mais aucune URL de proxy trouvée dans PROXIES.")

        if not verify_ssl:
            logger.warning(
                "Vérification SSL désactivée (verify_ssl=False) pour les appels à l'API Mistral. "
                "À réserver à un environnement de confiance."
            )
        http_client_kwargs['verify'] = verify_ssl

        custom_http_client = httpx.Client(**http_client_kwargs)
        self.client = Mistral(api_key=self.api_key, client=custom_http_client)

        # Statistiques
        self.stats: Dict[str, Any] = {
            'total_events': 0,
            'total_tokens': 0,
            'total_time': 0,
            'errors': 0,
            'batch_size': 0,
        }

    # ------------------------------------------------------------------
    # Génération des embeddings
    # ------------------------------------------------------------------

    def creer_embeddings(
        self,
        textes: List[str],
        batch_size: Optional[int] = None,
        delay: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Crée des embeddings pour une liste de textes.

        Les textes sont envoyés par batchs à l'API Mistral. En cas
        d'erreur sur un batch, jusqu'à `MAX_RETRIES` tentatives sont
        effectuées avec un délai croissant (backoff exponentiel) avant
        d'abandonner ce batch : les textes concernés reçoivent alors un
        vecteur nul et sont consignés dans la liste d'erreurs retournée.

        Args:
            textes (list[str]): Liste des textes à vectoriser. Les valeurs
                manquantes (`None`, `NaN`) sont traitées comme des chaînes
                vides.
            batch_size (int, optionnel): Nombre de textes par batch
                (défaut : `config.settings.EMBEDDING_BATCH_SIZE`).
            delay (float, optionnel): Délai entre batches en secondes
                (défaut : `config.settings.EMBEDDING_DELAY`).

        Returns:
            tuple[np.ndarray, list[dict]]: Un tuple `(embeddings, errors)` où
                `embeddings` est un tableau numpy de forme
                `(len(textes), dimension)`, et `errors` la liste des
                textes n'ayant pas pu être vectorisés (avec leur index et
                le message d'erreur). Si `textes` est vide, un tableau de
                forme `(0, 0)` est renvoyé.

        Raises:
            RuntimeError: Si tous les batches échouent (impossible de
                déterminer la dimension des embeddings — aucun vecteur
                n'a pu être calculé). Vérifiez la clé API, le modèle et la
                connectivité réseau.
        """
        from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_DELAY

        if not textes:
            logger.warning("Liste de textes vide : aucun embedding à générer.")
            self.stats['total_events'] = 0
            return np.empty((0, 0), dtype=np.float32), []

        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        delay = delay or EMBEDDING_DELAY

        # Sanitisation : les valeurs manquantes (None, NaN) deviennent des
        # chaînes vides plutôt que de faire échouer l'appel API.
        textes_propres = [t if isinstance(t, str) else ('' if t is None or t != t else str(t)) for t in textes]
        nb_vides = sum(1 for t in textes_propres if not t.strip())
        if nb_vides:
            logger.warning(
                "%d texte(s) vide(s) ou manquant(s) parmi les %d à vectoriser.",
                nb_vides, len(textes_propres)
            )

        logger.info(
            "Génération des embeddings (modèle=%s, %d textes, batchs de %d)",
            self.model, len(textes_propres), batch_size
        )

        embeddings: List[Optional[List[float]]] = []
        errors: List[Dict] = []
        emplacements_en_erreur: List[int] = []

        self.stats['total_events'] = len(textes_propres)
        self.stats['batch_size'] = batch_size

        start_time = time.time()
        total_batches = (len(textes_propres) + batch_size - 1) // batch_size

        for i in range(0, len(textes_propres), batch_size):
            batch = textes_propres[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info("Batch %d/%d (%d textes)...", batch_num, total_batches, len(batch))

            batch_embeddings, batch_errors = self._embed_batch_avec_retry(batch, offset=i)

            if batch_embeddings is not None:
                embeddings.extend(batch_embeddings)
                if self.dimension is None and batch_embeddings:
                    self.dimension = len(batch_embeddings[0])
            else:
                self.stats['errors'] += len(batch)
                errors.extend(batch_errors)
                # Vecteur nul temporaire : corrigé plus bas une fois la
                # dimension connue (elle peut ne pas encore l'être si ce
                # batch a échoué avant tout succès).
                for _ in batch:
                    emplacements_en_erreur.append(len(embeddings))
                    embeddings.append(None)

            if i + batch_size < len(textes_propres):
                time.sleep(delay)

        self.stats['total_time'] = time.time() - start_time

        if self.dimension is None:
            raise RuntimeError(
                "Impossible de déterminer la dimension des embeddings : tous les batches ont "
                f"échoué. Vérifiez la clé API, le modèle configuré ('{self.model}') et la "
                "connectivité réseau (voir les erreurs ci-dessus)."
            )

        for idx in emplacements_en_erreur:
            embeddings[idx] = [0.0] * self.dimension

        embeddings_array = np.array(embeddings, dtype=np.float32)

        logger.info(
            "Génération terminée en %.2fs (%d embeddings, dimension %d)",
            self.stats['total_time'], len(embeddings), self.dimension
        )
        if self.stats['errors'] > 0:
            logger.warning(
                "%d texte(s) en erreur (vecteurs nuls insérés à la place) — voir errors.json",
                self.stats['errors']
            )

        return embeddings_array, errors

    def _embed_batch_avec_retry(
        self, batch: List[str], offset: int
    ) -> Tuple[Optional[List[List[float]]], List[Dict]]:
        """
        Vectorise un batch de textes, avec nouvelles tentatives automatiques
        en cas d'erreur.

        Args:
            batch (list[str]): Textes du batch à vectoriser.
            offset (int): Index, dans la liste complète des textes, du
                premier élément de ce batch (utilisé pour reporter des
                erreurs avec le bon index global).

        Returns:
            tuple:
                - `(embeddings, [])` si le batch a été traité avec succès
                  (éventuellement après une ou plusieurs tentatives).
                - `(None, errors)` si toutes les tentatives ont échoué,
                  avec `errors` la liste des erreurs (une par texte du
                  batch).
        """
        derniere_erreur: Optional[Exception] = None

        for tentative in range(1, MAX_RETRIES + 1):
            try:
                batch_start = time.time()
                response = self.client.embeddings.create(model=self.model, inputs=batch)
                batch_time = time.time() - batch_start

                batch_embeddings = [item.embedding for item in response.data]

                if hasattr(response, 'usage') and response.usage:
                    self.stats['total_tokens'] += response.usage.total_tokens

                logger.info("  ✓ batch traité en %.2fs (tentative %d/%d)", batch_time, tentative, MAX_RETRIES)
                return batch_embeddings, []

            except Exception as e:
                derniere_erreur = e
                logger.warning("  ✗ échec tentative %d/%d : %s", tentative, MAX_RETRIES, e)
                if tentative < MAX_RETRIES:
                    attente = RETRY_BACKOFF_BASE * (2 ** (tentative - 1))
                    time.sleep(attente)

        errors = [
            {
                'index': offset + j,
                'texte': (texte[:100] + '...') if len(texte) > 100 else texte,
                'erreur': str(derniere_erreur),
            }
            for j, texte in enumerate(batch)
        ]
        return None, errors

    def creer_embeddings_depuis_dataframe(
        self,
        df: pd.DataFrame,
        colonne_texte: str = 'texte_complet',
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Crée des embeddings depuis un DataFrame pandas.

        Args:
            df (pd.DataFrame): DataFrame contenant les événements. Doit
                contenir la colonne `colonne_texte`, idéalement `id`,
                `titre`, `ville`, `date_debut` et `url`.
            colonne_texte (str, optionnel): Nom de la colonne contenant le
                texte à vectoriser. Défaut : `'texte_complet'`.
            batch_size (int, optionnel): Taille des batches (voir
                :meth:`creer_embeddings`).

        Returns:
            tuple[np.ndarray, list[dict], list[dict]]: `(embeddings, metadata, errors)`.
            `metadata` contient, pour chaque ligne de `df` (dans le même
            ordre), un dictionnaire avec les champs `id`, `titre`, `ville`,
            `date_debut`, `url` et `index` (position dans `df`).

        Raises:
            KeyError: Si `colonne_texte` n'existe pas dans `df`.
            RuntimeError: Voir :meth:`creer_embeddings`.
        """
        if colonne_texte not in df.columns:
            raise KeyError(
                f"La colonne '{colonne_texte}' est absente du DataFrame. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

        logger.info(
            "Préparation des données depuis le DataFrame (%d événements, colonne '%s')",
            len(df), colonne_texte
        )

        textes = df[colonne_texte].tolist()

        metadata = []
        for idx, row in df.iterrows():
            metadata.append({
                'id': self._valeur_ou_defaut(row, 'id', str(idx)),
                'titre': self._valeur_ou_defaut(row, 'titre'),
                'ville': self._valeur_ou_defaut(row, 'ville'),
                'date_debut': self._valeur_ou_defaut(row, 'date_debut'),
                'url': self._valeur_ou_defaut(row, 'url'),
                'index': int(idx) if isinstance(idx, (int, np.integer)) else idx,
            })

        embeddings, errors = self.creer_embeddings(textes, batch_size=batch_size)

        return embeddings, metadata, errors

    @staticmethod
    def _valeur_ou_defaut(row: pd.Series, col: str, defaut: str = '') -> str:
        """
        Retourne `row[col]` sous forme de chaîne, ou `defaut` si la colonne
        est absente ou la valeur manquante (NaN).

        Cette fonction évite deux pièges courants avec des DataFrames
        rechargés depuis un CSV : une conversion `int()` qui échoue sur un
        identifiant non numérique (ex: un slug), et une valeur `NaN`
        (float) là où une chaîne vide était attendue.

        Args:
            row (pd.Series): Ligne du DataFrame.
            col (str): Nom de la colonne à lire.
            defaut (str, optionnel): Valeur renvoyée si absente/NaN.

        Returns:
            str: Valeur sous forme de chaîne, ou `defaut`.
        """
        if col not in row or pd.isna(row[col]):
            return defaut
        return str(row[col])

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def sauvegarder_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        output_dir: str,
        errors: Optional[List[Dict]] = None,
    ) -> None:
        """
        Sauvegarde les embeddings et métadonnées sur disque.

        Fichiers générés dans `output_dir` :
            - `embeddings.npy` : tableau numpy des vecteurs.
            - `metadata.json` : métadonnées associées à chaque vecteur
              (même ordre que `embeddings`).
            - `stats_embeddings.json` : statistiques de génération.
            - `errors.json` : uniquement si `errors` est non vide.

        Args:
            embeddings (np.ndarray): Array numpy des embeddings.
            metadata (list[dict]): Liste des métadonnées, alignée avec
                `embeddings` (même longueur, même ordre).
            output_dir (str): Dossier de sortie (créé si besoin).
            errors (list[dict], optionnel): Liste des erreurs rencontrées.

        Returns:
            None

        Raises:
            ValueError: Si `embeddings` et `metadata` n'ont pas la même longueur.
            OSError: Si l'écriture sur disque échoue (droits, disque plein...).
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Incohérence : {len(embeddings)} embeddings pour {len(metadata)} métadonnées "
                "— les deux doivent être alignés (même ordre, même longueur)."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Sauvegarde des embeddings dans %s", output_path)

        try:
            embeddings_file = output_path / 'embeddings.npy'
            np.save(embeddings_file, embeddings)
            logger.info(
                "Embeddings sauvegardés : %s (%.2f Mo)",
                embeddings_file, embeddings_file.stat().st_size / 1024 / 1024
            )

            metadata_file = output_path / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(
                "Métadonnées sauvegardées : %s (%.2f Ko)",
                metadata_file, metadata_file.stat().st_size / 1024
            )

            stats_file = output_path / 'stats_embeddings.json'
            stats_complete = {
                **self.stats,
                'dimension': self.dimension or 0,
                'nombre_embeddings': len(embeddings),
                'modele': self.model,
                'erreurs_details': errors if errors else [],
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_complete, f, ensure_ascii=False, indent=2)
            logger.info("Statistiques sauvegardées : %s", stats_file)

            if errors:
                errors_file = output_path / 'errors.json'
                with open(errors_file, 'w', encoding='utf-8') as f:
                    json.dump(errors, f, ensure_ascii=False, indent=2)
                logger.warning("Erreurs sauvegardées : %s (%d erreurs)", errors_file, len(errors))

        except OSError as e:
            logger.error("Échec de la sauvegarde des embeddings dans %s : %s", output_path, e)
            raise

    def charger_embeddings(self, input_dir: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Charge des embeddings précédemment sauvegardés.

        Args:
            input_dir (str): Dossier contenant `embeddings.npy` et `metadata.json`.

        Returns:
            tuple[np.ndarray, list[dict]]: `(embeddings, metadata)`.

        Raises:
            FileNotFoundError: Si l'un des deux fichiers attendus est absent.
        """
        input_path = Path(input_dir)
        embeddings_file = input_path / 'embeddings.npy'
        metadata_file = input_path / 'metadata.json'

        if not embeddings_file.exists():
            raise FileNotFoundError(f"Fichier d'embeddings introuvable : {embeddings_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Fichier de métadonnées introuvable : {metadata_file}")

        embeddings = np.load(embeddings_file)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        dimension = embeddings.shape[1] if embeddings.ndim > 1 else 0
        logger.info(
            "Embeddings chargés depuis %s (%d vecteurs, dimension %d)",
            input_path, len(embeddings), dimension
        )

        return embeddings, metadata

    def obtenir_statistiques(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de génération accumulées.

        Returns:
            dict: Copie du dictionnaire de statistiques interne, avec les
            clés `total_events`, `total_tokens`, `total_time`, `errors`,
            `batch_size`.
        """
        return self.stats.copy()