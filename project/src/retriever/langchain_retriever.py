"""
LangChain BaseRetriever qui encapsule EventRetriever
"""

from typing import List
from pydantic import ConfigDict, BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from project.src.retriever.event_retriever import EventRetriever


class LangChainEventRetriever(BaseRetriever):
    """
    Retriever LangChain qui encapsule EventRetriever pour intégration LCEL.

    Conserve l'index Faiss et le format de métadonnées existants.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_retriever: EventRetriever
    """EventRetriever sous-jacent qui gère l'index Faiss"""

    k: int = 5
    """Nombre d'événements à récupérer"""

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Récupère les documents pertinents via EventRetriever.

        Args:
            query: Question de l'utilisateur

        Returns:
            Liste de Documents LangChain
        """
        events = self.event_retriever.retrieve_with_scores(query, k=self.k)

        documents = []
        for event in events:
            page_content = self._format_event_text(event)
            documents.append(
                Document(
                    page_content=page_content,
                    metadata=event
                )
            )

        return documents

    def _format_event_text(self, event: dict) -> str:
        """
        Formate un événement en texte pour le page_content du Document.

        Args:
            event: Dict contenant les champs de l'événement

        Returns:
            Texte formaté
        """
        parts = []

        if titre := event.get('titre'):
            parts.append(f"Titre: {titre}")
        if description := event.get('description'):
            parts.append(f"Description: {description}")
        if ville := event.get('ville'):
            parts.append(f"Ville: {ville}")
        if date_debut := event.get('date_debut'):
            parts.append(f"Date de début: {date_debut}")
        if date_fin := event.get('date_fin'):
            parts.append(f"Date de fin: {date_fin}")
        if lieu_nom := event.get('lieu_nom'):
            parts.append(f"Lieu: {lieu_nom}")
        if categorie := event.get('categorie'):
            parts.append(f"Catégorie: {categorie}")

        return "\n".join(parts)


def load_langchain_retriever(
    index_dir: str = "project/data/vectorstore",
    mistral_api_key: str = None,
    k: int = 5
) -> LangChainEventRetriever:
    """
    Charge le retriever LangChain avec les chemins par défaut.

    Args:
        index_dir: Répertoire contenant l'index et les métadonnées
        mistral_api_key: Clé API Mistral
        k: Nombre d'événements à récupérer

    Returns:
        LangChainEventRetriever initialisé
    """
    from project.src.retriever.event_retriever import load_retriever

    event_retriever = load_retriever(index_dir, mistral_api_key)

    return LangChainEventRetriever(
        event_retriever=event_retriever,
        k=k
    )