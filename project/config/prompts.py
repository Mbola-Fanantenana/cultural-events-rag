"""
Templates de prompts pour le RAG
"""

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans les événements culturels en France.

Ton rôle est d'aider les utilisateurs à trouver des événements pertinents en te basant UNIQUEMENT sur les informations fournies.

Règles importantes :
- Réponds de manière claire, concise et structurée
- Base-toi UNIQUEMENT sur les événements fournis dans le contexte
- Si aucun événement ne correspond, dis-le clairement
- Mentionne toujours les informations clés : titre, ville, date
- Si l'utilisateur demande des détails, fournis-les depuis le contexte
- Ne jamais inventer d'informations
"""


def build_rag_prompt(query: str, events: list) -> str:
    """
    Construit le prompt pour le RAG
    
    Args:
        query: Question de l'utilisateur
        events: Liste d'événements pertinents
    
    Returns:
        Prompt formaté
    """
    # Construire le contexte
    if not events:
        context = "Aucun événement pertinent trouvé."
    else:
        context_parts = []
        for i, event in enumerate(events, 1):
            event_text = f"""Événement {i}:
Titre: {event.get('titre', 'N/A')}
Description: {event.get('description', 'N/A')}
Ville: {event.get('ville', 'N/A')}
Date de début: {event.get('date_debut', 'N/A')}
Date de fin: {event.get('date_fin', 'N/A')}
Lieu: {event.get('lieu_nom', 'N/A')}
Catégorie: {event.get('categorie', 'N/A')}
"""
            # Ajouter le score si disponible
            if 'similarity_score' in event:
                event_text += f"Score de pertinence: {event['similarity_score']:.2f}\n"
            
            context_parts.append(event_text)
        
        context = "\n".join(context_parts)
    
    # Construire le prompt complet
    prompt = f"""{SYSTEM_PROMPT}

CONTEXTE - Événements pertinents :

{context}

QUESTION DE L'UTILISATEUR :
{query}

RÉPONSE :
Réponds à la question en te basant uniquement sur les événements ci-dessus."""
    
    return prompt


def build_simple_prompt(query: str, events: list) -> str:
    """
    Version simplifiée du prompt (pour tests)
    
    Args:
        query: Question de l'utilisateur
        events: Liste d'événements pertinents
    
    Returns:
        Prompt formaté
    """
    if not events:
        return f"Question: {query}\n\nAucun événement pertinent trouvé."
    
    context = "\n\n".join([
        f"{i+1}. {e['titre']} - {e['ville']} - {e['date_debut']}"
        for i, e in enumerate(events)
    ])
    
    return f"""Voici des événements culturels :

{context}

Question : {query}

Réponds de manière concise."""
