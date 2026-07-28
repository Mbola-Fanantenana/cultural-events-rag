"""
Classe pour gérer l'index FAISS.

Ce module fournit la classe FAISSIndex permettant de créer, gérer,
sauvegarder et charger un index FAISS pour la recherche de similarité
vectorielle dans le cadre du pipeline RAG.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class FAISSIndex:
    """
    Gère l'index FAISS pour la recherche de similarité vectorielle.

    Cette classe encapsule les opérations de création, recherche,
    mise à jour, sauvegarde et chargement d'un index FAISS.

    Attributes:
        dimension (int): Dimension des vecteurs d'embeddings.
        index (faiss.Index): Index FAISS actif.
        metadata (list): Métadonnées associées aux vecteurs indexés.
    """

    def __init__(self, dimension: int = 1024):
        """
        Initialise l'index FAISS.

        Args:
            dimension (int): Dimension des vecteurs. Par défaut 1024
                (dimension utilisée par Mistral Embed).
        """
        self.dimension = dimension
        self.index = None
        self.metadata = None

    def creer_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "IndexFlatL2"
    ) -> faiss.Index:
        """
        Crée un index FAISS à partir des embeddings fournis.

        Args:
            embeddings (np.ndarray): Array numpy des embeddings de forme
                (N, dimension).
            index_type (str): Type d'index FAISS à créer. Valeurs possibles :
                - "IndexFlatL2"  : Recherche exacte par distance L2
                  (recommandé pour < 1M vecteurs).
                - "IndexFlatIP"  : Recherche exacte par produit scalaire.
                - "IndexIVFFlat" : Recherche approximative avec quantification.
                - "IndexHNSW"    : Recherche approximative très rapide.

        Returns:
            faiss.Index: L'index FAISS créé et peuplé.

        Raises:
            ValueError: Si la dimension des embeddings ne correspond pas
                à self.dimension, ou si le type d'index est inconnu.
        """
        print(f"\n🗂️ Création de l'index FAISS")
        print(f"   Type      : {index_type}")
        print(f"   Dimension : {self.dimension}")
        print(f"   Vecteurs  : {len(embeddings)}")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension des embeddings ({embeddings.shape[1]}) "
                f"ne correspond pas à la dimension attendue ({self.dimension})."
            )

        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)

        elif index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)

        elif index_type == "IndexIVFFlat":
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            print(f"   Entraînement IVF avec {nlist} clusters...")
            self.index.train(embeddings)

        elif index_type == "IndexHNSW":
            M = 32
            self.index = faiss.IndexHNSWFlat(self.dimension, M)

        else:
            raise ValueError(f"Type d'index non supporté : {index_type}")

        print(f"   Ajout des vecteurs à l'index...")
        self.index.add(embeddings)

        print(f"✓ Index créé — {self.index.ntotal} vecteurs indexés.")
        return self.index

    def ajouter_vecteurs(
        self,
        nouveaux_embeddings: np.ndarray,
        nouvelles_metadata: List[Dict]
    ) -> None:
        """
        Ajoute de nouveaux vecteurs à un index existant (mise à jour incrémentale).

        Cette méthode permet d'enrichir l'index sans le recréer entièrement.
        Elle met également à jour la liste des métadonnées en mémoire.

        Args:
            nouveaux_embeddings (np.ndarray): Array numpy des nouveaux vecteurs
                de forme (N, dimension).
            nouvelles_metadata (List[Dict]): Liste des métadonnées associées
                aux nouveaux vecteurs. Doit avoir la même longueur que
                nouveaux_embeddings.

        Raises:
            ValueError: Si l'index n'est pas initialisé, si la dimension
                des nouveaux vecteurs est incorrecte, ou si les longueurs
                des embeddings et métadonnées ne correspondent pas.
        """
        if self.index is None:
            raise ValueError(
                "Index non initialisé. Appelez d'abord creer_index()."
            )

        if nouveaux_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension des nouveaux embeddings ({nouveaux_embeddings.shape[1]}) "
                f"ne correspond pas à la dimension de l'index ({self.dimension})."
            )

        if len(nouveaux_embeddings) != len(nouvelles_metadata):
            raise ValueError(
                f"Le nombre de vecteurs ({len(nouveaux_embeddings)}) "
                f"ne correspond pas au nombre de métadonnées "
                f"({len(nouvelles_metadata)})."
            )

        nb_avant = self.index.ntotal
        self.index.add(nouveaux_embeddings)

        if self.metadata is not None:
            self.metadata.extend(nouvelles_metadata)
        else:
            self.metadata = nouvelles_metadata

        print(
            f"✓ {len(nouveaux_embeddings)} vecteurs ajoutés "
            f"({nb_avant} → {self.index.ntotal})."
        )

    def reconstruire_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        index_type: str = "IndexFlatL2"
    ) -> None:
        """
        Reconstruit entièrement l'index à partir de nouveaux embeddings.

        Utile lorsque des suppressions logiques ont été effectuées dans
        les métadonnées et qu'une reconstruction propre est nécessaire.

        Args:
            embeddings (np.ndarray): Nouveaux embeddings complets de forme
                (N, dimension).
            metadata (List[Dict]): Nouvelles métadonnées complètes.
            index_type (str): Type d'index FAISS à recréer.

        Raises:
            ValueError: Si les dimensions ou longueurs sont incohérentes.
        """
        print("\n🔄 Reconstruction complète de l'index...")
        self.index = None
        self.metadata = None
        self.creer_index(embeddings, index_type=index_type)
        self.metadata = metadata
        print(f"✓ Index reconstruit avec {self.index.ntotal} vecteurs.")

    def valider_coherence(self) -> bool:
        """
        Vérifie la cohérence entre l'index FAISS et les métadonnées en mémoire.

        Contrôle que le nombre de vecteurs dans l'index correspond au nombre
        d'entrées dans les métadonnées.

        Returns:
            bool: True si l'index et les métadonnées sont cohérents,
                False sinon.

        Raises:
            ValueError: Si l'index ou les métadonnées ne sont pas chargés.
        """
        if self.index is None:
            raise ValueError("Index non initialisé.")
        if self.metadata is None:
            raise ValueError("Métadonnées non chargées.")

        nb_index = self.index.ntotal
        nb_meta = len(self.metadata)
        coherent = nb_index == nb_meta

        if coherent:
            print(f"✓ Cohérence validée : {nb_index} vecteurs / {nb_meta} métadonnées.")
        else:
            print(
                f"⚠️ Incohérence détectée : "
                f"{nb_index} vecteurs dans l'index, "
                f"{nb_meta} entrées dans les métadonnées."
            )

        return coherent

    def rechercher(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recherche les k vecteurs les plus proches dans l'index.

        Args:
            query_embedding (np.ndarray): Vecteur de requête de forme
                (dimension,) ou (1, dimension).
            k (int): Nombre de résultats à retourner.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Un tuple (distances, indices) où :
                - distances : Array de forme (1, k) avec les distances.
                - indices   : Array de forme (1, k) avec les indices des
                  vecteurs les plus proches.

        Raises:
            ValueError: Si l'index n'est pas initialisé ou si la dimension
                du vecteur de requête est incorrecte.
        """
        if self.index is None:
            raise ValueError(
                "Index non initialisé. Appelez d'abord creer_index()."
            )

        if query_embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Dimension du vecteur de requête ({query_embedding.shape[-1]}) "
                f"ne correspond pas à la dimension de l'index ({self.dimension})."
            )

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def rechercher_avec_metadata(
        self,
        query_embedding: np.ndarray,
        metadata: List[Dict],
        k: int = 5
    ) -> List[Dict]:
        """
        Recherche les k vecteurs les plus proches et retourne leurs métadonnées.

        Args:
            query_embedding (np.ndarray): Vecteur de requête.
            metadata (List[Dict]): Liste complète des métadonnées indexées.
            k (int): Nombre de résultats à retourner.

        Returns:
            List[Dict]: Liste de dictionnaires contenant pour chaque résultat :
                - rank (int)     : Rang du résultat (1 = plus proche).
                - distance (float): Distance L2 par rapport à la requête.
                - score (float)  : Score de similarité calculé comme
                  1 / (1 + distance).
                - Les champs de métadonnées associés au vecteur.

        Raises:
            ValueError: Si l'index n'est pas initialisé ou si la dimension
                est incorrecte (propagé depuis rechercher()).
        """
        distances, indices = self.rechercher(query_embedding, k)

        resultats = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):
                resultat = {
                    'rank': i + 1,
                    'distance': float(dist),
                    'score': float(1 / (1 + dist)),
                    **metadata[idx]
                }
                resultats.append(resultat)

        return resultats

    def sauvegarder(
        self,
        output_dir: str,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Sauvegarde l'index FAISS, les métadonnées et les statistiques sur disque.

        Args:
            output_dir (str): Chemin du dossier de destination. Créé
                automatiquement s'il n'existe pas.
            metadata (Optional[List[Dict]]): Métadonnées à sauvegarder.
                Si None, les métadonnées en mémoire (self.metadata) sont
                utilisées si disponibles.

        Raises:
            ValueError: Si l'index n'est pas initialisé.
            OSError: Si le dossier de destination ne peut pas être créé
                ou si l'écriture des fichiers échoue.
        """
        if self.index is None:
            raise ValueError(
                "Index non initialisé. Appelez d'abord creer_index()."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Sauvegarde de l'index FAISS dans : {output_path}")

        # Sauvegarde de l'index FAISS
        index_file = output_path / 'faiss.index'
        faiss.write_index(self.index, str(index_file))
        print(f"   ✓ Index FAISS    : {index_file} "
              f"({index_file.stat().st_size / 1024 / 1024:.2f} Mo)")

        # Sauvegarde des métadonnées
        meta_a_sauvegarder = metadata if metadata is not None else self.metadata
        if meta_a_sauvegarder is not None:
            metadata_file = output_path / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(meta_a_sauvegarder, f, ensure_ascii=False, indent=2)
            print(f"   ✓ Métadonnées   : {metadata_file} "
                  f"({metadata_file.stat().st_size / 1024:.2f} Ko)")
            self.metadata = meta_a_sauvegarder

        # Sauvegarde des statistiques
        stats = self.obtenir_statistiques()
        stats_file = output_path / 'stats_index.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Statistiques  : {stats_file}")

    def charger(self, input_dir: str) -> Tuple[faiss.Index, Optional[List[Dict]]]:
        """
        Charge un index FAISS et ses métadonnées depuis le disque.

        Args:
            input_dir (str): Chemin du dossier contenant les fichiers
                'faiss.index' et optionnellement 'metadata.json'.

        Returns:
            Tuple[faiss.Index, Optional[List[Dict]]]: Un tuple contenant :
                - L'index FAISS chargé.
                - La liste des métadonnées, ou None si absentes.

        Raises:
            FileNotFoundError: Si le fichier 'faiss.index' est introuvable
                dans le dossier spécifié.
        """
        input_path = Path(input_dir)

        index_file = input_path / 'faiss.index'
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index FAISS introuvable : {index_file}"
            )

        self.index = faiss.read_index(str(index_file))

        metadata_file = input_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        print(f"\n✓ Index FAISS chargé depuis : {input_path}")
        print(f"   Vecteurs : {self.index.ntotal}")
        print(f"   Type     : {type(self.index).__name__}")

        return self.index, self.metadata

    def obtenir_statistiques(self) -> Dict:
        """
        Retourne les statistiques courantes de l'index.

        Returns:
            Dict: Dictionnaire contenant :
                - dimension (int)         : Dimension des vecteurs.
                - nombre_vecteurs (int)   : Nombre de vecteurs indexés.
                - type_index (str)        : Nom de la classe de l'index FAISS.
                - is_trained (bool)       : Indique si l'index est entraîné.
                - nb_metadata (int|None)  : Nombre d'entrées de métadonnées
                  en mémoire, ou None si non chargées.
                - coherent (bool|None)    : True si index et métadonnées sont
                  cohérents, None si métadonnées absentes.

        Raises:
            ValueError: Si l'index n'est pas initialisé (retourne un dict vide).
        """
        if self.index is None:
            return {}

        nb_meta = len(self.metadata) if self.metadata is not None else None
        coherent = (
            self.index.ntotal == nb_meta
            if nb_meta is not None
            else None
        )

        return {
            'dimension': self.dimension,
            'nombre_vecteurs': self.index.ntotal,
            'type_index': type(self.index).__name__,
            'is_trained': (
                self.index.is_trained
                if hasattr(self.index, 'is_trained')
                else True
            ),
            'nb_metadata': nb_meta,
            'coherent': coherent,
        }
