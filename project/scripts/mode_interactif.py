import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from project.src.rag.rag_pipeline import load_rag_pipeline


def main():
    
    print("\n" + "="*60)
    print("DÉMONSTRATION - PIPELINE RAG")
    print("="*60)
    print("\nChargement du système...")
    
    # Charge le pipeline
    load_dotenv()
    rag = load_rag_pipeline()
    
    print("\n✓ Système prêt !")
    print("\nVous pouvez maintenant poser vos questions.")
    print("Tapez 'quit' pour quitter.\n")
    
    # Lance le mode interactif
    rag.interactive_mode()


if __name__ == "__main__":
    main()
