"""
Module de pré-traitement (preprocessing) des données d'événements OpenAgenda.

Ce module fournit la classe :class:`DataCleaner`, responsable de :
    - l'extraction des champs clés depuis des données JSON brutes,
    - la structuration de ces données dans un DataFrame pandas,
    - le nettoyage (dédoublonnage, filtrage, nettoyage HTML, normalisation
      des espaces),
    - le calcul de statistiques descriptives sur le jeu de données nettoyé.

Ce module constitue la première étape du pipeline RAG (Retrieval-Augmented
Generation) : les données produites ici (colonne ``texte_complet``) sont
destinées à être vectorisées dans une étape ultérieure (voir le script de
vectorisation).

Exemple d'utilisation:
    >>> cleaner = DataCleaner(events)
    >>> df = cleaner.pipeline()
    >>> stats = cleaner.obtenir_statistiques()
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes attendues dans le DataFrame produit par extraire_champs_cles().
# Centralisées ici pour garantir un schéma stable même si `events` est vide.
COLONNES_ATTENDUES = [
    'id', 'titre', 'description', 'description_longue', 'ville',
    'adresse', 'region', 'date_debut', 'date_fin', 'categories',
    'url', 'texte_complet',
]


class DataCleaner:
    """
    Classe responsable du nettoyage et de la structuration des données d'événements.

    Cette classe transforme des données brutes (JSON, telles que renvoyées par
    l'API OpenAgenda) en un DataFrame exploitable pour des tâches d'analyse
    ou de RAG (Retrieval-Augmented Generation).

    Attributes:
        events (list[dict]): Données brutes des événements.
        df (pd.DataFrame | None): Données transformées. ``None`` tant que
            :meth:`extraire_champs_cles` n'a pas été appelée.
    """

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        """
        Initialise le DataCleaner.

        Args:
            events (list[dict]): Liste des événements bruts (issus du JSON
                OpenAgenda). Chaque élément doit être un dictionnaire ; les
                éléments malformés sont ignorés (avec un avertissement loggé)
                lors de l'extraction.

        Raises:
            TypeError: Si `events` n'est pas une liste.
        """
        if not isinstance(events, list):
            raise TypeError(f"`events` doit être une liste, reçu : {type(events).__name__}")

        self.events = events
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extraire_champs_cles(self) -> pd.DataFrame:
        """
        Extrait et structure les champs essentiels des événements.

        Les champs extraits incluent :
            - Informations textuelles (titre, description, description longue)
            - Localisation (ville, adresse, région)
            - Dates (début, fin)
            - Catégories
            - URL

        Les événements dont l'élément n'est pas un dictionnaire sont ignorés
        et signalés via un avertissement (log).

        Returns:
            pd.DataFrame: DataFrame contenant les données structurées prêtes
            à être nettoyées (colonnes : voir :data:`COLONNES_ATTENDUES`).
            Un DataFrame vide mais avec le bon schéma de colonnes est renvoyé
            si ``self.events`` est vide.
        """
        donnees_structurees = []

        for i, event in enumerate(self.events):
            if not isinstance(event, dict):
                logger.warning("Événement à l'index %d ignoré : type inattendu (%s)", i, type(event).__name__)
                continue

            # Extraction des textes multilingues
            titre = self._extraire_texte(event.get('title', {}))
            description = self._extraire_texte(event.get('description', {}))
            long_description = self._extraire_texte(event.get('longDescription', {}))

            # Extraction de la localisation
            location = event.get('location', {}) or {}
            ville = location.get('city', '')
            adresse = location.get('address', '')
            region = location.get('region', '')

            # Extraction des mots-clés/catégories
            keywords = event.get('keywords', {})
            if isinstance(keywords, dict):
                keywords = keywords.get('fr', [])
            categories = ', '.join(keywords) if keywords else ''

            # Extraction des dates
            date_debut = self._extraire_date(event.get('firstTiming', {}), 'begin')
            date_fin = self._extraire_date(event.get('lastTiming', {}), 'end')

            # Construction de l'URL
            url = self._construire_url(event)

            # Création de l'enregistrement
            record = {
                'id': event.get('uid', ''),
                'titre': titre,
                'description': description,
                'description_longue': long_description,
                'ville': ville,
                'adresse': adresse,
                'region': region,
                'date_debut': date_debut,
                'date_fin': date_fin,
                'categories': categories,
                'url': url,
                'texte_complet': self._creer_texte_complet(
                    titre, description, long_description, ville, adresse,
                    categories, date_debut, date_fin
                ),
            }

            donnees_structurees.append(record)

        if donnees_structurees:
            self.df = pd.DataFrame(donnees_structurees)
        else:
            # On force le schéma de colonnes pour que nettoyer_donnees()
            # ne lève pas de KeyError sur un DataFrame vide.
            self.df = pd.DataFrame(columns=COLONNES_ATTENDUES)
            logger.warning("Aucun événement valide trouvé : DataFrame vide créé.")

        return self.df

    def _extraire_texte(self, field: Union[Dict[str, str], str, None]) -> str:
        """
        Extrait le texte d'un champ multilingue.

        Args:
            field (dict | str | None): Dictionnaire multilingue
                (ex: ``{"fr": "...", "en": "..."}``) ou chaîne simple.

        Returns:
            str: Texte extrait, en priorité en français, puis en anglais,
            puis dans la première langue disponible. Chaîne vide si `field`
            est vide, ``None``, ou ne contient aucune valeur exploitable.
        """
        if isinstance(field, dict):
            # Priorité : français, puis anglais, puis première langue disponible
            return field.get('fr', field.get('en', next(iter(field.values()), '')))
        return str(field) if field else ''

    def _extraire_date(self, timing_dict: Optional[Dict[str, str]], key: str) -> str:
        """
        Extrait et formate une date depuis un objet timing.

        Gère les dates au format ISO 8601, avec ou sans fuseau horaire et
        avec ou sans millisecondes, par exemple :
            - ``"2026-04-08T14:00:00.000+02:00"``
            - ``"2026-04-08T14:00:00+02:00"``
            - ``"2026-04-08T14:00:00"``

        Args:
            timing_dict (dict | None): Dictionnaire contenant les clés
                ``begin``/``end`` (typiquement ``firstTiming``/``lastTiming``).
            key (str): Clé à extraire, ``'begin'`` ou ``'end'``.

        Returns:
            str: Date formatée ``"YYYY-MM-DD HH:MM:SS"``. Si le format n'a
            pas pu être reconnu, la chaîne brute est renvoyée telle quelle.
            Chaîne vide si la date est absente.
        """
        if not timing_dict or key not in timing_dict:
            return ''

        date_str = timing_dict.get(key, '')
        if not date_str:
            return ''

        # Normalisation : suppression des millisecondes (".000") si présentes
        date_str_clean = date_str.replace('.000', '')

        # On essaie d'abord avec fuseau horaire, puis sans
        formats = ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S']
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str_clean, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

        logger.warning("Format de date non reconnu pour la clé '%s' : %s", key, date_str)
        return date_str

    def _construire_url(self, event: Dict[str, Any]) -> str:
        """
        Construit l'URL de l'événement sur OpenAgenda.

        Args:
            event (dict): Dictionnaire de l'événement, doit contenir
                idéalement ``slug`` et ``originAgenda.slug``.

        Returns:
            str: URL complète de la forme
            ``https://openagenda.com/{agenda_slug}/events/{slug}``.
            Si ``originAgenda`` est absent, un fallback
            ``https://openagenda.com/events/{slug}`` est utilisé.
            Chaîne vide si aucun ``slug`` n'est disponible.
        """
        slug = event.get('slug', '')
        origin_agenda = event.get('originAgenda', {}) or {}
        agenda_slug = origin_agenda.get('slug', '')

        if slug and agenda_slug:
            return f"https://openagenda.com/{agenda_slug}/events/{slug}"
        elif slug:
            # Fallback si pas d'agenda d'origine
            return f"https://openagenda.com/events/{slug}"
        else:
            return ''

    def _creer_texte_complet(
        self,
        titre: str,
        description: str,
        long_description: str,
        ville: str,
        adresse: str,
        categories: str,
        date_debut: str = '',
        date_fin: str = '',
    ) -> str:
        """
        Crée un texte complet et lisible, destiné à l'embedding (vectorisation).

        Le texte est construit en assemblant les champs disponibles, chacun
        préfixé par une étiquette (ex: "Titre:", "Description:"), séparés par
        des sauts de ligne. Les champs vides ou redondants (ex: description
        longue identique à la description courte) sont omis.

        Args:
            titre (str): Titre de l'événement.
            description (str): Description courte.
            long_description (str): Description longue/détaillée.
            ville (str): Ville de l'événement.
            adresse (str): Adresse de l'événement.
            categories (str): Catégories/mots-clés, séparés par des virgules.
            date_debut (str, optionnel): Date de début formatée.
            date_fin (str, optionnel): Date de fin formatée.

        Returns:
            str: Texte complet formaté, prêt à être vectorisé.
        """
        parties = []

        if titre:
            parties.append(f"Titre: {titre}")

        if description:
            parties.append(f"Description: {description}")

        if long_description and long_description != description:
            parties.append(f"Détails: {long_description}")

        # Localisation
        localisation = []
        if ville:
            localisation.append(ville)
        if adresse:
            localisation.append(adresse)

        if localisation:
            parties.append(f"Lieu: {', '.join(localisation)}")

        # Dates
        if date_debut:
            parties.append(f"Date de début: {date_debut}")
        if date_fin and date_fin != date_debut:
            parties.append(f"Date de fin: {date_fin}")

        if categories:
            parties.append(f"Catégories: {categories}")

        return "\n".join(parties)

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    def nettoyer_donnees(self) -> pd.DataFrame:
        """
        Nettoie et filtre les données précédemment extraites.

        Étapes réalisées, dans l'ordre :
            1. Suppression des doublons (basée sur ``id``).
            2. Suppression des événements sans titre.
            3. Suppression des événements sans description.
            4. Suppression des descriptions trop courtes (< 50 caractères).
            5. Suppression des événements sans ville.
            6. Normalisation des espaces dans les champs texte.
            7. Nettoyage du HTML dans les descriptions.
            8. Reconstruction du champ ``texte_complet`` après nettoyage.
            9. Réinitialisation de l'index du DataFrame.

        Returns:
            pd.DataFrame: DataFrame nettoyé et filtré.

        Raises:
            ValueError: Si :meth:`extraire_champs_cles` n'a pas été appelée
                au préalable (``self.df`` est ``None``).
        """
        if self.df is None:
            raise ValueError("Appelez d'abord extraire_champs_cles()")

        nb_initial = len(self.df)
        logger.info("Nettoyage des données... (nombre initial : %d)", nb_initial)

        if nb_initial == 0:
            logger.warning("DataFrame vide : rien à nettoyer.")
            return self.df

        # 1. Supprimer les doublons basés sur l'ID
        avant = len(self.df)
        self.df = self.df.drop_duplicates(subset=['id'], keep='first')
        apres = len(self.df)
        if avant != apres:
            logger.info("%d doublons supprimés", avant - apres)

        # 2. Filtrer les événements avec titre manquant ou vide
        avant = len(self.df)
        self.df = self.df[
            (self.df['titre'].notna()) &
            (self.df['titre'].str.strip() != '')
        ]
        apres = len(self.df)
        if avant != apres:
            logger.info("%d événements sans titre supprimés", avant - apres)

        # 3. Filtrer les événements avec description manquante ou vide
        avant = len(self.df)
        self.df = self.df[
            (self.df['description'].notna()) &
            (self.df['description'].str.strip() != '')
        ]
        apres = len(self.df)
        if avant != apres:
            logger.info("%d événements sans description supprimés", avant - apres)

        # 4. Filtrer les descriptions trop courtes (< 50 caractères)
        avant = len(self.df)
        self.df = self.df[self.df['description'].str.len() >= 50]
        apres = len(self.df)
        if avant != apres:
            logger.info("%d descriptions trop courtes supprimées", avant - apres)

        # 5. Filtrer les événements sans ville
        avant = len(self.df)
        self.df = self.df[
            (self.df['ville'].notna()) &
            (self.df['ville'].str.strip() != '')
        ]
        apres = len(self.df)
        if avant != apres:
            logger.info("%d événements sans ville supprimés", avant - apres)

        # 6. Nettoyer les espaces en trop
        colonnes_texte = ['titre', 'description', 'description_longue',
                           'ville', 'adresse', 'region', 'categories']

        for col in colonnes_texte:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
                self.df[col] = self.df[col].str.strip()

        # 7. Nettoyer le HTML dans les descriptions (si présent)
        self.df['description'] = self.df['description'].apply(self._nettoyer_html)
        self.df['description_longue'] = self.df['description_longue'].apply(self._nettoyer_html)

        # 8. Recréer le texte complet après nettoyage
        self.df['texte_complet'] = self.df.apply(
            lambda row: self._creer_texte_complet(
                row['titre'],
                row['description'],
                row['description_longue'],
                row['ville'],
                row['adresse'],
                row['categories'],
                row['date_debut'],
                row['date_fin']
            ),
            axis=1
        )

        # 9. Réinitialiser l'index
        self.df = self.df.reset_index(drop=True)

        logger.info("Nombre final : %d", len(self.df))
        logger.info("%d événements supprimés au total", nb_initial - len(self.df))

        return self.df

    def _nettoyer_html(self, texte: Optional[str]) -> Optional[str]:
        """
        Nettoie le HTML d'un texte : suppression des balises et décodage des
        entités HTML courantes (y compris les caractères accentués français).

        Args:
            texte (str | None): Texte potentiellement au format HTML.
                Peut être ``NaN`` (valeur manquante pandas) ou vide.

        Returns:
            str | None: Texte nettoyé (balises supprimées, entités décodées,
            espaces multiples réduits, espaces de bord retirés). La valeur
            d'entrée est renvoyée telle quelle si elle est ``NaN`` ou vide.
        """
        if pd.isna(texte) or texte == '':
            return texte

        # Supprimer les balises HTML
        texte = re.sub(r'<[^>]+>', '', texte)

        # Décoder les entités HTML courantes
        entites = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&eacute;': 'é',
            '&egrave;': 'è',
            '&ecirc;': 'ê',
            '&agrave;': 'à',
            '&acirc;': 'â',
            '&ccedil;': 'ç',
            '&ocirc;': 'ô',
            '&ucirc;': 'û',
            '&icirc;': 'î',
        }

        for entite, caractere in entites.items():
            texte = texte.replace(entite, caractere)

        # Supprimer les espaces multiples
        texte = re.sub(r'\s+', ' ', texte)

        return texte.strip()

    # ------------------------------------------------------------------
    # Statistiques & pipeline
    # ------------------------------------------------------------------

    def obtenir_statistiques(self) -> Dict[str, Any]:
        """
        Calcule des statistiques descriptives sur les données nettoyées.

        Returns:
            dict: Dictionnaire vide si ``self.df`` est ``None`` ou vide,
            sinon un dictionnaire contenant les clés suivantes :
                - ``nombre_total`` (int) : nombre total d'événements.
                - ``nombre_villes`` (int) : nombre de villes distinctes.
                - ``top_villes`` (dict) : top 10 des villes les plus fréquentes.
                - ``longueur_moyenne_description`` (int) : longueur moyenne
                  (en caractères) du champ ``description``.
                - ``longueur_moyenne_texte_complet`` (int) : longueur moyenne
                  du champ ``texte_complet``.
                - ``taux_avec_categories`` (float) : pourcentage d'événements
                  ayant au moins une catégorie renseignée.
                - ``taux_avec_description_longue`` (float) : pourcentage
                  d'événements ayant une description longue.
                - ``taux_avec_date_debut`` (float) : pourcentage d'événements
                  ayant une date de début renseignée.
                - ``taux_avec_url`` (float) : pourcentage d'événements ayant
                  une URL renseignée.
        """
        if self.df is None or len(self.df) == 0:
            return {}

        stats = {
            'nombre_total': len(self.df),
            'nombre_villes': self.df['ville'].nunique(),
            'top_villes': self.df['ville'].value_counts().head(10).to_dict(),
            'longueur_moyenne_description': int(self.df['description'].str.len().mean()),
            'longueur_moyenne_texte_complet': int(self.df['texte_complet'].str.len().mean()),
            'taux_avec_categories': round((self.df['categories'] != '').sum() / len(self.df) * 100, 1),
            'taux_avec_description_longue': round((
                (self.df['description_longue'].notna()) &
                (self.df['description_longue'] != '')
            ).sum() / len(self.df) * 100, 1),
            'taux_avec_date_debut': round((
                (self.df['date_debut'].notna()) &
                (self.df['date_debut'] != '')
            ).sum() / len(self.df) * 100, 1),
            'taux_avec_url': round((
                (self.df['url'].notna()) &
                (self.df['url'] != '')
            ).sum() / len(self.df) * 100, 1),
        }

        return stats

    def pipeline(self) -> pd.DataFrame:
        """
        Exécute l'ensemble du pipeline de pré-traitement en une seule fois :
        extraction puis nettoyage.

        Cette méthode est le point d'entrée recommandé pour un usage en
        pipeline (et facilite l'écriture de tests unitaires de bout en bout).

        Returns:
            pd.DataFrame: DataFrame final, extrait et nettoyé.

        Example:
            >>> cleaner = DataCleaner(events)
            >>> df = cleaner.pipeline()
        """
        self.extraire_champs_cles()
        self.nettoyer_donnees()
        return self.df


if __name__ == "__main__":
    # Exemple d'utilisation en ligne de commande / débogage rapide.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    exemple_events = [
        {
            "uid": "1",
            "title": {"fr": "Concert de jazz"},
            "description": {"fr": "Un super concert de jazz en plein air, avec des artistes locaux et internationaux."},
            "location": {"city": "Antananarivo", "address": "Rue de la Musique"},
            "keywords": {"fr": ["musique", "jazz"]},
            "firstTiming": {"begin": "2026-08-01T20:00:00.000+03:00"},
            "lastTiming": {"end": "2026-08-01T23:00:00.000+03:00"},
            "slug": "concert-jazz",
            "originAgenda": {"slug": "mon-agenda"},
        }
    ]

    cleaner = DataCleaner(exemple_events)
    df_final = cleaner.pipeline()
    print(df_final)
    print(cleaner.obtenir_statistiques())