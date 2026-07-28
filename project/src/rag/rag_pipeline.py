import time
from typing import Dict, List
from project.src.retriever.event_retriever import EventRetriever
from project.src.retriever.langchain_retriever import LangChainEventRetriever
from project.src.generator.response_generator import ResponseGenerator


class RAGPipeline:

    def __init__(
        self,
        retriever: EventRetriever,
        generator: ResponseGenerator,
        default_k: int = 5
    ):
        """
        Args:
            retriever: EventRetriever pour la recherche
            generator: ResponseGenerator pour la génération
            default_k: Nombre d'événements à récupérer par défaut
        """
        self._event_retriever = retriever
        self.retriever = LangChainEventRetriever(
            event_retriever=retriever,
            k=default_k
        )
        self.generator = generator
        self.default_k = default_k

    def query(
        self,
        question: str,
        k: int = None,
        return_sources: bool = True
    ) -> Dict:
        if k is None:
            k = self.default_k

        self.retriever.k = k

        start_time = time.time()

        retrieval_start = time.time()
        docs = self.retriever.invoke(question)
        retrieval_time = time.time() - retrieval_start

        events = [doc.metadata for doc in docs]

        generation_start = time.time()
        answer = self.generator.generate(question, events)
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        result = {
            'question': question,
            'answer': answer,
            'nb_events_retrieved': len(events),
            'timing': {
                'retrieval_ms': round(retrieval_time * 1000, 2),
                'generation_ms': round(generation_time * 1000, 2),
                'total_ms': round(total_time * 1000, 2)
            }
        }

        if return_sources:
            result['source_events'] = events

        return result

    def batch_query(
        self,
        questions: List[str],
        k: int = None
    ) -> List[Dict]:
        """
        Traite plusieurs questions.

        Args:
            questions: Liste de questions
            k: Nombre d'événements à récupérer

        Returns:
            Liste de résultats
        """
        results = []
        for question in questions:
            result = self.query(question, k, return_sources=False)
            results.append(result)

        return results

    def interactive_mode(self):
        print("Tapez 'quit' pour quitter\n")

        while True:
            question = input("Question : ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print("\n Recherche en cours...")
            result = self.query(question)

            print("\n" + "-"*60)
            print("RÉPONSE :")
            print("-"*60)
            print(result['answer'])

            print("\n" + "-"*60)
            print("MÉTADONNÉES :")
            print("-"*60)
            print(f"Événements récupérés : {result['nb_events_retrieved']}")
            print(f"Temps de recherche : {result['timing']['retrieval_ms']} ms")
            print(f"Temps de génération : {result['timing']['generation_ms']} ms")
            print(f"Temps total : {result['timing']['total_ms']} ms")

            if result.get('source_events'):
                print("\n" + "-"*60)
                print("ÉVÉNEMENTS SOURCES :")
                print("-"*60)
                for i, event in enumerate(result['source_events'], 1):
                    print(f"{i}. {event.get('titre', 'N/A')} - {event.get('ville', 'N/A')}")
                    print(f"   Score : {event.get('similarity_score', 'N/A'):.2f}")

            print("\n" + "="*60 + "\n")


def load_rag_pipeline(
    index_dir: str = "project/data/vectorstore",
    mistral_api_key: str = None,
    default_k: int = 5
) -> RAGPipeline:
    """
    Charge le pipeline RAG complet.

    Args:
        index_dir: Répertoire de l'index
        mistral_api_key: Clé API Mistral
        default_k: Nombre d'événements par défaut

    Returns:
        RAGPipeline initialisé
    """
    from project.src.retriever.event_retriever import load_retriever
    from project.src.generator.response_generator import load_generator

    retriever = load_retriever(index_dir, mistral_api_key)
    generator = load_generator(mistral_api_key)

    return RAGPipeline(retriever, generator, default_k)