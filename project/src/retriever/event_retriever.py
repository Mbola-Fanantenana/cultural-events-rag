import numpy as np
import faiss
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from mistralai import Mistral
import httpx


class EventRetriever:
    
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        mistral_api_key: str,
        model: str = "mistral-embed"
    ):
        """
        Args:
            index_path: Chemin vers l'index FAISS
            metadata_path: Chemin vers les métadonnées
            mistral_api_key: Clé API Mistral
            model: Modèle d'embedding à utiliser
        """
        # get l'index FAISS
        self.index = faiss.read_index(index_path)
        print(f"Index FAISS chargé : {self.index.ntotal} vecteurs")
                    
        # get métadonnées
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"Métadonnées chargées : {len(self.metadata)} événements")
        
        # CONFIGURATION DU PROXY
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        
        if http_proxy or https_proxy:
            if http_proxy and not http_proxy.startswith(("http://", "https://")):
                http_proxy = f"http://{http_proxy}"
            if https_proxy and not https_proxy.startswith(("http://", "https://")):
                https_proxy = f"http://{https_proxy}"
            
            custom_client = httpx.Client(
                proxies={
                    "http://": http_proxy,
                    "https://": https_proxy
                },
                verify=False  
            )
            
            self.client = Mistral(
                api_key=mistral_api_key,
                client=custom_client
            )
        else:
            self.client = Mistral(api_key=mistral_api_key)
        
        self.model = model  
    
    def _create_query_embedding(self, query: str) -> np.ndarray:
        """
        Crée l'embedding d'une question
        
        Args:
            query: Question de l'utilisateur
        
        Returns:
            Vecteur d'embedding (1, 1024)
        """
        response = self.client.embeddings.create(
            model=self.model,
            inputs=[query]
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding.reshape(1, -1)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_distances: bool = False
    ) -> List[Dict] | Tuple[List[Dict], np.ndarray]:
        """
        Récupère les k événements les plus pertinents
        
        Args:
            query: Question de l'utilisateur
            k: Nombre d'événements à récupérer
            return_distances: Si True, retourne aussi les distances
        """
        query_embedding = self._create_query_embedding(query)
        
        distances, indices = self.index.search(query_embedding, k)
        
        # get métadonnées
        events = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                event = self.metadata[idx].copy()  # ← Créer une copie
                events.append(event)
        
        if return_distances:
            return events, distances[0]
        
        return events
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict]:
        """
        Récupère les événements avec leurs scores de similarité
        
        Args:
            query: Question de l'utilisateur
            k: Nombre d'événements à récupérer
        
        Returns:
            Liste d'événements avec scores de similarité
        """
        # Récupérer les événements et les distances
        events, distances = self.retrieve(query, k, return_distances=True)
        
        # Convertir les distances L2 en scores de similarité (0-1)
        # Score = 1 / (1 + distance)
        # Plus la distance est petite, plus le score est élevé
        scores = 1 / (1 + distances)
        
        # Ajouter les scores aux événements
        for event, score in zip(events, scores):
            event['similarity_score'] = float(score)
        
        return events

    
def load_retriever(
    index_dir: str = "project/data/vectorstore",
    mistral_api_key: str = None
) -> EventRetriever:
    """
    Charge le retriever avec les chemins par défaut
    
    Args:
        index_dir: Répertoire contenant l'index et les métadonnées
        mistral_api_key: Clé API Mistral
    
    Returns:
        EventRetriever initialisé
    """
    index_path = Path(index_dir) / "faiss.index"
    metadata_path = Path(index_dir) / "metadata.json"
    
    if mistral_api_key is None:
        from dotenv import load_dotenv
        load_dotenv()
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    return EventRetriever(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        mistral_api_key=mistral_api_key
    )
