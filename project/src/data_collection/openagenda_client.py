"""
Client HTTP pour l'API OpenAgenda v2.

Ce module fournit la classe :class:`OpenAgendaClient`, responsable de
toutes les interactions avec l'API publique OpenAgenda :
    - Récupération de la liste des agendas disponibles.
    - Récupération des événements d'un agenda donné.

Il constitue la couche d'accès aux données brutes du pipeline RAG,
utilisée par le script `scripts/scraping.py`.

Références:
    Documentation API OpenAgenda v2 :
    https://developers.openagenda.com/10-lecture/
"""

import logging
import requests

from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    Timeout,
    HTTPError,
    RequestException,
)
from typing import List, Dict, Optional, Tuple

from config.settings import (
    OPENAGENDA_API_KEY,
    OPENAGENDA_BASE_URL,
    PROXIES_REQUESTS,
)

logger = logging.getLogger(__name__)

# Délai d'attente par défaut pour les requêtes HTTP (en secondes)
DEFAULT_TIMEOUT: int = 30


class OpenAgendaClient:
    """
    Client HTTP pour l'API OpenAgenda v2.

    Encapsule les appels HTTP vers l'API OpenAgenda avec gestion des
    erreurs, du proxy et du timeout. Chaque méthode retourne une liste
    vide en cas d'erreur non critique pour ne pas interrompre le pipeline.

    Attributes:
        api_key (str): Clé API OpenAgenda utilisée pour l'authentification.
        base_url (str): URL de base de l'API OpenAgenda.
        headers (dict): En-têtes HTTP envoyés avec chaque requête.
        proxies (dict): Configuration du proxy HTTP/HTTPS.
        timeout (int): Délai d'attente maximum par requête (en secondes).

    Example:
        >>> client = OpenAgendaClient()
        >>> agendas = client.get_agendas(limite=10)
        >>> events = client.get_events_from_agenda(agendas[0]['uid'], limite=20)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialise le client OpenAgenda.

        Args:
            api_key (str, optionnel): Clé API OpenAgenda. Si absente,
                lue depuis `config.settings.OPENAGENDA_API_KEY` (elle-même
                définie via la variable d'environnement `OPENAGENDA_API_KEY`).
            base_url (str, optionnel): URL de base de l'API. Si absente,
                lue depuis `config.settings.OPENAGENDA_BASE_URL`.
            timeout (int, optionnel): Délai d'attente maximum pour chaque
                requête HTTP, en secondes. Défaut : 30.

        Raises:
            ValueError: Si aucune clé API n'est disponible (ni en paramètre,
                ni dans la configuration).
        """
        self.api_key = api_key or OPENAGENDA_API_KEY
        self.base_url = base_url or OPENAGENDA_BASE_URL
        self.timeout = timeout
        self.proxies = PROXIES_REQUESTS

        if not self.api_key:
            raise ValueError(
                "Clé API OpenAgenda manquante. "
                "Définissez OPENAGENDA_API_KEY dans le fichier .env."
            )

        # L'authentification OpenAgenda se fait via l'en-tête 'key'
        self.headers: Dict[str, str] = {"key": self.api_key}

        logger.info(
            "Client OpenAgenda initialisé (base_url=%s, timeout=%ds).",
            self.base_url, self.timeout,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Méthodes publiques
    # ─────────────────────────────────────────────────────────────────────

    def get_agendas(
        self,
        limite: int = 50,
        after: list = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Récupère une page d'agendas depuis l'API OpenAgenda.

        La pagination est gérée via le curseur `after` retourné par l'API
        dans chaque réponse. Ce curseur est une liste de deux éléments
        [index, uid] à passer tel quel dans la requête suivante.

        Args:
            limite (int, optionnel): Nombre d'agendas par page. Défaut : 50.
            after (list, optionnel): Curseur de pagination retourné par
                l'appel précédent, sous la forme [index, uid]. Si None,
                la requête démarre depuis le début de la liste.

        Returns:
            Tuple[List[Dict], Dict]: Un tuple contenant :
                - La liste des agendas de la page courante.
                - La réponse brute complète (contient le curseur `after`
                pour la page suivante et le `total`).

        Raises:
            ValueError: Si `limite` est inférieur ou égal à 0.
        """
        if limite <= 0:
            raise ValueError(
                f"Le paramètre 'limite' doit être strictement positif "
                f"(reçu : {limite})."
            )

        url = f"{self.base_url}/agendas"
        params: Dict = {"size": limite}

        if after:
            # L'API OpenAgenda attend deux paramètres séparés :
            # after[]=<index>&after[]=<uid>
            # requests accepte une liste de tuples pour les clés dupliquées
            params = [
                ("size", limite),
                ("after[]", after[0]),
                ("after[]", after[1]),
            ]
            logger.info(
                "Requête GET %s (size=%d, after[]=%s)...",
                url, limite, after,
            )
        else:
            params = [("size", limite)]
            logger.info(
                "Requête GET %s (size=%d, after=None — début de liste)...",
                url, limite,
            )

        response = self._get(url, params=params)
        if response is None:
            return [], {}

        data = response.json()
        agendas = data.get("agendas", [])
        total = data.get("total", 0)
        curseur_suivant = data.get("after")

        logger.info(
            "%d agenda(s) reçu(s) (total API=%d, prochain curseur=%s).",
            len(agendas), total, curseur_suivant,
        )

        return agendas, data


    def get_events_from_agenda(
        self,
        agenda_uid: str,
        limite: int = 50,
    ) -> List[Dict]:
        """
        Récupère les événements d'un agenda spécifique.

        Effectue une requête GET sur l'endpoint
        `/agendas/{agenda_uid}/events` et retourne la liste des événements.

        Args:
            agenda_uid (str): Identifiant unique (UID) de l'agenda dont
                on souhaite récupérer les événements.
            limite (int, optionnel): Nombre maximum d'événements à
                récupérer. Défaut : 50. Correspond au paramètre `size`
                de l'API.

        Returns:
            List[Dict]: Liste des événements retournés par l'API. Chaque
                événement est un dictionnaire contenant les informations
                détaillées de l'événement (titre, dates, lieu, etc.).
                Retourne une liste vide si l'agenda est introuvable,
                en cas d'erreur HTTP ou réseau.

        Raises:
            ValueError: Si `agenda_uid` est vide ou None, ou si `limite`
                est inférieur ou égal à 0.
        """
        if not agenda_uid:
            raise ValueError(
                "Le paramètre 'agenda_uid' ne peut pas être vide ou None."
            )
        if limite <= 0:
            raise ValueError(
                f"Le paramètre 'limite' doit être strictement positif (reçu : {limite})."
            )

        url = f"{self.base_url}/agendas/{agenda_uid}/events"
        params = {"size": limite, "detailed": 1}

        logger.debug(
            "Requête GET %s (size=%d, detailed=1)...", url, limite
        )

        response = self._get(url, params=params)
        if response is None:
            return []

        events = response.json().get("events", [])
        logger.debug(
            "Agenda uid=%s : %d événement(s) reçu(s).", agenda_uid, len(events)
        )

        return events

    # ─────────────────────────────────────────────────────────────────────
    # Méthodes privées
    # ─────────────────────────────────────────────────────────────────────

    def _get(
        self,
        url: str,
        params: Optional[Dict] = None,
    ) -> Optional[requests.Response]:
        """
        Effectue une requête HTTP GET avec gestion centralisée des erreurs.

        Toutes les erreurs réseau et HTTP sont interceptées et journalisées.
        La méthode retourne None en cas d'erreur pour permettre aux
        appelants de retourner une liste vide sans interrompre le pipeline.

        Args:
            url (str): URL complète de la ressource à requêter.
            params (dict, optionnel): Paramètres de la requête GET
                (query string).

        Returns:
            requests.Response: Objet réponse HTTP si la requête a réussi
                (code 2xx).
            None: En cas d'erreur réseau, timeout, ou code HTTP non 2xx.
        """
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                proxies=self.proxies,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response

        except Timeout:
            logger.error(
                "Timeout après %ds lors de la requête GET %s.", self.timeout, url
            )
        except RequestsConnectionError as e:
            logger.error(
                "Erreur de connexion lors de la requête GET %s : %s", url, e
            )
        except HTTPError as e:
            logger.error(
                "Erreur HTTP %s lors de la requête GET %s : %s",
                e.response.status_code if e.response else "?", url, e,
            )
        except RequestException as e:
            logger.error(
                "Erreur réseau inattendue lors de la requête GET %s : %s", url, e
            )

        return None
