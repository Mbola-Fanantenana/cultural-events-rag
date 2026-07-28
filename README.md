# 📚 RAG Project - Système de recherche augmentée par génération

## 🧠 Présentation

Ce projet implémente un système de **RAG (Retrieval-Augmented Generation)**
permettant de poser des questions en langage naturel sur un ensemble de données
d'événements culturels collectés via l'API **OpenAgenda**.

Le système combine :
- une **collecte automatique et versionnée** des événements (OpenAgenda API)
- une **recherche vectorielle** rapide (FAISS)
- un modèle de **génération de texte** (Mistral AI)
- un **pipeline complet** de traitement de données

---

## 🎯 Objectifs

- Collecter et versionner des données d’événements culturels (OpenAgenda)
- Nettoyer et structurer les données brutes
- Générer des embeddings vectoriels (Mistral Embed)
- Construire et maintenir un index FAISS pour la recherche sémantique
- Implémenter un système RAG pour répondre aux questions en langage naturel
- Évaluer la qualité des réponses générées

---

## ⚙️ Installation et reproduction

### 1. Cloner le projet

```bash
git clone <url_du_repo>
cd rag_project
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d’environnement

Créez un fichier `.env` à la racine du projet :

```env
OPENAGENDA_API_KEY=votre_clé_openagenda
MISTRAL_API_KEY=votre_clé_mistral
```

---

## 🚀 Utilisation du pipeline

Les scripts sont à exécuter dans l'ordre suivant depuis la racine du projet.

### 1. Collecte des données

```bash
python project/scripts/scraping.py
```

> Collecte une page de 20 agendas et leurs événements via l'API OpenAgenda.
> À chaque exécution, le curseur de pagination est automatiquement avancé
> pour explorer de nouveaux agendas. Les données sont fusionnées et
> dédupliquées avec les collectes précédentes.

### 2. Nettoyage des données

```bash
python project/scripts/nettoyage_data.py
```

> Nettoie et structure les données brutes collectées.

### 3. Génération des embeddings

```bash
python project/scripts/creer_embeddings.py
```

> Génère les embeddings vectoriels via l'API Mistral Embed.

### 4. Création de l'index FAISS

```bash
python project/scripts/creer_index_faiss.py
```

> Construit l'index de recherche vectorielle FAISS.

### 5. Lancer une requête RAG

```bash
python project/scripts/query_rag.py
```

> Pose une question en langage naturel et obtient une réponse générée
> par Mistral à partir des événements les plus pertinents.

### 6. Mode interactif (optionnel)

```bash
python project/scripts/mode_interactif.py
```

> Lance une session de questions-réponses en continu, sans relancer le script à chaque question.

---

## 🔄 Versionnement des données

Le système de collecte intègre un mécanisme de **versionnement automatique**
qui permet d'accumuler les données au fil des exécutions.

### Principe

```bash
Exécution 1   curseur=None        → agendas  1-20  → ~297 événements
Exécution 2   curseur=[1,336790]  → agendas 21-40  → ~550 événements
Exécution 3   curseur=[1,409846]  → agendas 41-60  → ~820 événements
...
Exécution N   fin de liste        → reset curseur  → recommence depuis le début
```

### Fichiers générés

| Fichier | Description |
|---|---|
| `data/raw/evenements_bruts.json` | Cumul fusionné et dédupliqué de toutes les collectes |
| `data/raw/versions/evenements_bruts_YYYYMMDD_HHMMSS.json` | Snapshot horodaté de chaque collecte |
| `data/raw/collecte_log.json` | Journal de toutes les sessions avec statistiques |
| `data/raw/collecte_offset.json` | Curseur de pagination pour la prochaine session |

### Déduplication

Chaque événement est identifié par son champ `uid` OpenAgenda.
Lors de la fusion :
- Un événement **absent** est ajouté.
- Un événement **modifié** (même uid, contenu différent) est mis à jour.
- Un événement **identique** est ignoré (doublon).

---

## 🏗️ Architecture du projet


```bash
rag_project/
│
├── .env                        # Variables d'environnement (non versionné)
├── requirements.txt            # Dépendances Python
├── README.md
│
└── project/
    │
    ├── config/                 # Configuration centralisée
    │   ├── settings.py         # Paramètres globaux (API, chemins, modèles)
    │   └── prompts.py          # Templates de prompts RAG
    │
    ├── data/                   # Données du projet (non versionnées)
    │   ├── raw/                # Données brutes OpenAgenda
    │   │   ├── evenements_bruts.json       # Cumul fusionné courant
    │   │   ├── collecte_log.json           # Journal des collectes
    │   │   ├── collecte_offset.json        # Curseur de pagination
    │   │   └── versions/                   # Snapshots horodatés
    │   ├── processed/          # Données nettoyées
    │   │   ├── evenements_nettoyes.json
    │   │   └── versions/
    │   ├── embeddings/         # Embeddings vectoriels
    │   │   ├── embeddings.npy
    │   │   ├── metadata.json
    │   │   └── versions/
    │   └── vectorstore/        # Index FAISS
    │       ├── faiss.index
    │       └── versions/
    │
    ├── scripts/                # Scripts exécutables du pipeline
    │   ├── scraping.py         # Collecte et versionnement des données
    │   ├── nettoyage_data.py   # Nettoyage des données brutes
    │   ├── creer_embeddings.py # Génération des embeddings
    │   ├── creer_index_faiss.py # Construction de l'index FAISS
    │   └── query_rag.py        # Interface de requête RAG
    │
    └── src/                    # Code source principal
        ├── data_collection/    # Collecte et versionnement
        │   ├── openagenda_client.py  # Client HTTP OpenAgenda API
        |   ├── data_cleaner.py       # Nettoyage des données brutes
        │   └── data_versioner.py     # Versionnement et fusion des données
        ├── embeddings/         # Génération des embeddings
        ├── retriever/
        │   ├── event_retriever.py # Recherche vectorielle FAISS + embedding
        │   └── langchain_retriever.py # Wrapper Langchain autour d'EventRetriever
        ├── generator/
        │   └── response_generator.py # Génération de réponse via chaîne LCEL
        ├── rag/
        │   └── rag_pipeline.py # Orchestration retriever -> generator
        ├── vectorstore/        # Gestion de l'index FAISS
        └── evaluator/          # Évaluation de la qualité des réponses
```

---

## 🔄 Fonctionnement du système RAG

```
┌─────────────────────────────────────────────────────────┐
│                     PIPELINE DE DONNÉES                  │
│                                                          │
│  OpenAgenda API → Collecte → Nettoyage → Embeddings      │
│       (paginée)    (versionnée)           (Mistral)      │
│                                    ↓                     │
│                              Index FAISS                 │
└─────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────┐
│              PIPELINE RAG (orchestré via LangChain)      │
│                                                          │
│  Question utilisateur                                    │
│       ↓                                                  │
│  Embedding de la question (Mistral Embed)                │
│       ↓                                                  │
│  LangChainEventRetriever (BaseRetriever LangChain)        │
│  → recherche des K documents les plus proches (FAISS)    │
│       ↓                                                  │
│  Construction du prompt (contexte + question)             │
│       ↓                                                  │
│  Chaîne LCEL : PromptTemplate | ChatMistralAI | Parser    │
│       ↓                                                  │
│  Réponse en langage naturel                               │
└─────────────────────────────────────────────────────────┘
```

### Choix d'intégration LangChain

L'index FAISS et le format de métadonnées custom sont conservés tels quels
(pas de migration vers `langchain_community.vectorstores.FAISS`, qui imposerait
un format de sérialisation différent et une réindexation complète). L'intégration
se fait via un `BaseRetriever` LangChain (`LangChainEventRetriever`) qui encapsule
le retriever existant, et une chaîne LCEL (`PromptTemplate | ChatMistralAI | StrOutputParser`)
pour la génération, connectée directement au client Mistral. Le prompt engineering
(`config/prompts.py::build_rag_prompt`) reste indépendant de LangChain et est injecté
tel quel dans la chaîne.

---

## 🛠️ Technologies utilisées

| Composant | Technologie |
|---|---|
| Collecte de données | API OpenAgenda v2 |
| Embeddings | Mistral Embed (`mistral-embed`) |
| Génération de texte | Mistral AI (`mistral-small-latest`) |
| Orchestration recherche → génération | LangChain (LCEL) — `langchain-core`, `langchain-mistralai` |
| Recherche vectorielle | FAISS (`IndexFlatL2`) |
| Versionnement des données | Système maison (curseur + fusion JSON) |
| Langage | Python 3.10+ |

---

## 🧪 Tests

Les tests unitaires sont disponibles dans le dossier `tests/` :

```bash
python -m pytest tests/
```

---

## 📝 Variables de configuration

Les principaux paramètres sont centralisés dans `project/config/settings.py` :

| Variable | Description | Valeur par défaut |
|---|---|---|
| `OPENAGENDA_NB_AGENDAS` | Agendas collectés par session | `20` |
| `OPENAGENDA_EVENTS_PAR_AGENDA` | Événements max par agenda | `30` |
| `MISTRAL_MODEL` | Modèle de génération | `mistral-small-latest` |
| `MISTRAL_EMBEDDING_MODEL` | Modèle d'embedding | `mistral-embed` |
| `FAISS_INDEX_TYPE` | Type d'index FAISS | `IndexFlatL2` |
| `FAISS_TOP_K` | Documents récupérés par requête | `5` |
| `EMBEDDING_DIMENSION` | Dimension des vecteurs | `1024` |