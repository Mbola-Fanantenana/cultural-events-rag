"""
Vérification rapide du format after[] corrigé.

Usage:
    python scripts/verif_curseur.py
"""

import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OPENAGENDA_API_KEY, OPENAGENDA_BASE_URL, PROXIES_REQUESTS

def main():
    url = f"{OPENAGENDA_BASE_URL}/agendas"
    headers = {"key": OPENAGENDA_API_KEY}
    curseur = [1, 336790]

    # Format liste de tuples
    params = [
        ("size", 3),
        ("after[]", curseur[0]),
        ("after[]", curseur[1]),
    ]

    print(f"🧪 Test format liste de tuples...")
    response = requests.get(
        url,
        headers=headers,
        params=params,
        proxies=PROXIES_REQUESTS,
        timeout=30,
    )

    print(f"   Status  : {response.status_code}")
    print(f"   URL     : {response.url}")

    if response.status_code == 200:
        data = response.json()
        agendas = data.get("agendas", [])
        curseur_retour = data.get("after")
        print(f"   ✅ Succès — {len(agendas)} agendas")
        print(f"   Prochain curseur : {curseur_retour}")
        print(f"   Titres : {[a.get('title') for a in agendas]}")
    else:
        print(f"   ❌ Échec — {response.text[:200]}")

if __name__ == "__main__":
    main()
