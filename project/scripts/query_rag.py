
""" Script pour tester le pipeline RAG
"""
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
from dotenv import load_dotenv
from project.src.rag.rag_pipeline import load_rag_pipeline


def test_questions_simples():
    """Teste avec des questions simples"""
    
    print("\n" + "="*60)
    print("TEST 1 : Questions simples")
    print("="*60)
    
    # Charger le pipeline
    load_dotenv()
    rag = load_rag_pipeline()
    
    # Questions de test
    questions = [
        "Quels sont les événements à Châtellerault ?",
        "Y a-t-il des visites guidées ?",
        "Quels événements sont gratuits ?",
        "Que faire ce week-end à Châtellerault ?",
        "Y a-t-il des expositions ?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*60}")
        print(f"Question {i} : {question}")
        print('─'*60)
        
        result = rag.query(question, k=3)
        
        print(f"\nRéponse :\n{result['answer']}")
        print(f"\nÉvénements utilisés : {result['nb_events_retrieved']}")
        print(f"Temps total : {result['timing']['total_ms']} ms")


def test_questions_specifiques():
    """Teste avec des questions spécifiques"""
    
    print("\n" + "="*60)
    print("TEST 2 : Questions spécifiques")
    print("="*60)
    
    load_dotenv()
    rag = load_rag_pipeline()
    
    questions = [
        "Parle-moi du Carillon Bollée",
        "Quels sont les horaires de la Manu ?",
        "Y a-t-il des événements pour enfants ?",
        "Quels sont les événements en avril 2026 ?",
        "Où se trouve la Tamiserie ?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*60}")
        print(f"Question {i} : {question}")
        print('─'*60)
        
        result = rag.query(question, k=5)
        
        print(f"\nRéponse :\n{result['answer']}")
        
        # Afficher les événements sources
        print(f"\nÉvénements sources :")
        for j, event in enumerate(result['source_events'], 1):
            print(f"  {j}. {event['titre']} (score: {event['similarity_score']:.2f})")


def test_mode_interactif():
    """Lance le mode interactif"""
    
    load_dotenv()
    rag = load_rag_pipeline()
    
    rag.interactive_mode()


def main():
    """Fonction principale"""
    
    print("\n" + "="*60)
    print("ÉVALUATION DU PIPELINE RAG")
    print("="*60)
    
    # Test 1 : Questions simples
    test_questions_simples()
    
    # Test 2 : Questions spécifiques
    test_questions_specifiques()
    
    # Mode interactif
    print("\n" + "="*60)
    print("Voulez-vous tester en mode interactif ? (o/n)")
    print("="*60)
    
    choix = input("> ").strip().lower()
    if choix in ['o', 'oui', 'y', 'yes']:
        test_mode_interactif()


if __name__ == "__main__":
    main()
