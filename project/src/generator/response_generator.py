"""
Generator pour générer les réponses avec Mistral Chat via LangChain LCEL
"""

import os
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from project.config.prompts import build_rag_prompt


class ResponseGenerator:
    """Génère des réponses avec Mistral Chat via chaîne LCEL"""

    def __init__(
        self,
        mistral_api_key: str,
        model: str = "mistral-small-latest",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialise le generator avec LangChain.

        Args:
            mistral_api_key: Clé API Mistral
            model: Modèle de chat à utiliser
            temperature: Température (0-1, plus bas = plus déterministe)
            max_tokens: Nombre maximum de tokens dans la réponse
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = ChatMistralAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            mistral_api_key=mistral_api_key
        )

        self.prompt_template = PromptTemplate.from_template("{prompt}")
        self.output_parser = StrOutputParser()

        self.chain = self.prompt_template | self.llm | self.output_parser

    def generate(
        self,
        query: str,
        context_events: List[Dict],
        prompt_builder=build_rag_prompt
    ) -> str:
        """
        Génère une réponse basée sur le contexte via chaîne LCEL.

        Args:
            query: Question de l'utilisateur
            context_events: Événements pertinents
            prompt_builder: Fonction pour construire le prompt

        Returns:
            Réponse générée
        """
        prompt = prompt_builder(query, context_events)

        response = self.chain.invoke({"prompt": prompt})

        return response

    def generate_with_metadata(
        self,
        query: str,
        context_events: List[Dict]
    ) -> Dict:
        """
        Génère une réponse avec métadonnées.

        Args:
            query: Question de l'utilisateur
            context_events: Événements pertinents

        Returns:
            Dict avec réponse et métadonnées
        """
        answer = self.generate(query, context_events)

        return {
            'answer': answer,
            'nb_events_used': len(context_events),
            'model': self.model,
            'temperature': self.temperature
        }


def load_generator(mistral_api_key: str = None) -> ResponseGenerator:
    """
    Charge le generator avec les paramètres par défaut.

    Args:
        mistral_api_key: Clé API Mistral

    Returns:
        ResponseGenerator initialisé
    """
    if mistral_api_key is None:
        from dotenv import load_dotenv
        load_dotenv()
        mistral_api_key = os.getenv("MISTRAL_API_KEY")

    return ResponseGenerator(
        mistral_api_key=mistral_api_key,
        model="mistral-small-latest",
        temperature=0.3,
        max_tokens=1000
    )