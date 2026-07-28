"""
Tests unitaires pour la classe FAISSIndex.

Ce module teste les fonctionnalités principales de la classe FAISSIndex
définie dans src/vectorstore/faiss_index.py :
    - Création de l'index (différents types)
    - Ajout de vecteurs (mise à jour incrémentale)
    - Reconstruction de l'index
    - Validation de cohérence index / métadonnées
    - Recherche de vecteurs similaires
    - Sauvegarde et chargement sur disque
    - Statistiques

Usage:
    pytest tests/test_vectorstore.py -v
"""

import json
import tempfile
import pytest
import numpy as np

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.faiss_index import FAISSIndex


# ─────────────────────────────────────────────────────────────────────────────
# Constantes et Fixtures
# ─────────────────────────────────────────────────────────────────────────────

DIMENSION   = 64   # Dimension réduite pour accélérer les tests
NB_VECTEURS = 20   # Nombre de vecteurs de base


@pytest.fixture
def embeddings() -> np.ndarray:
    """
    Génère un tableau numpy d'embeddings aléatoires reproductibles.

    Returns:
        np.ndarray: Tableau de forme (NB_VECTEURS, DIMENSION), dtype float32.
    """
    np.random.seed(42)
    return np.random.rand(NB_VECTEURS, DIMENSION).astype("float32")


@pytest.fixture
def metadata() -> list:
    """
    Génère une liste de métadonnées fictives alignée avec les embeddings.

    Returns:
        list[dict]: Liste de NB_VECTEURS dictionnaires avec les clés
            'titre' et 'ville'.
    """
    return [
        {"titre": f"Événement {i}", "ville": f"Ville {i}"}
        for i in range(NB_VECTEURS)
    ]


@pytest.fixture
def index_cree(embeddings: np.ndarray, metadata: list) -> FAISSIndex:
    """
    Retourne un FAISSIndex initialisé, peuplé et avec métadonnées en mémoire.

    Args:
        embeddings: Fixture d'embeddings aléatoires.
        metadata: Fixture de métadonnées fictives.

    Returns:
        FAISSIndex: Instance prête à l'emploi pour les tests.
    """
    fi = FAISSIndex(dimension=DIMENSION)
    fi.creer_index(embeddings, index_type="IndexFlatL2")
    fi.metadata = metadata
    return fi


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Création de l'index
# ─────────────────────────────────────────────────────────────────────────────

class TestCreerIndex:
    """Tests de la méthode creer_index."""

    def test_creation_index_flat_l2(self, embeddings: np.ndarray):
        """
        Vérifie la création d'un index IndexFlatL2.

        L'index doit être non nul et contenir exactement NB_VECTEURS vecteurs.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexFlatL2")

        assert fi.index is not None
        assert fi.index.ntotal == NB_VECTEURS

    def test_creation_index_flat_ip(self, embeddings: np.ndarray):
        """
        Vérifie la création d'un index IndexFlatIP (produit scalaire).

        L'index doit contenir exactement NB_VECTEURS vecteurs.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexFlatIP")

        assert fi.index is not None
        assert fi.index.ntotal == NB_VECTEURS

    def test_creation_index_hnsw(self, embeddings: np.ndarray):
        """
        Vérifie la création d'un index IndexHNSW.

        L'index doit contenir exactement NB_VECTEURS vecteurs.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexHNSW")

        assert fi.index is not None
        assert fi.index.ntotal == NB_VECTEURS

    def test_creation_index_ivfflat(self, embeddings: np.ndarray):
        """
        Vérifie la création d'un index IndexIVFFlat (approximatif).

        L'index doit être entraîné et contenir exactement NB_VECTEURS vecteurs.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexIVFFlat")

        assert fi.index is not None
        assert fi.index.ntotal == NB_VECTEURS
        assert fi.index.is_trained is True

    def test_dimension_incorrecte_leve_value_error(self):
        """
        Vérifie qu'une ValueError est levée si la dimension des embeddings
        ne correspond pas à la dimension déclarée dans FAISSIndex.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        mauvais_embeddings = np.random.rand(5, DIMENSION + 10).astype("float32")

        with pytest.raises(ValueError, match="Dimension"):
            fi.creer_index(mauvais_embeddings)

    def test_type_index_inconnu_leve_value_error(self, embeddings: np.ndarray):
        """
        Vérifie qu'une ValueError est levée pour un type d'index non supporté.
        """
        fi = FAISSIndex(dimension=DIMENSION)

        with pytest.raises(ValueError, match="non supporté"):
            fi.creer_index(embeddings, index_type="IndexInconnu")

    def test_index_retourne_objet_faiss(self, embeddings: np.ndarray):
        """
        Vérifie que creer_index retourne bien un objet FAISS (non None).
        """
        fi = FAISSIndex(dimension=DIMENSION)
        index = fi.creer_index(embeddings, index_type="IndexFlatL2")

        assert index is not None
        assert index.ntotal == NB_VECTEURS


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Ajout de vecteurs (mise à jour incrémentale)
# ─────────────────────────────────────────────────────────────────────────────

class TestAjouterVecteurs:
    """Tests de la méthode ajouter_vecteurs."""

    def test_ajout_incremente_ntotal(self, index_cree: FAISSIndex):
        """
        Vérifie que l'ajout de N vecteurs incrémente ntotal de N.
        """
        np.random.seed(99)
        nouveaux = np.random.rand(5, DIMENSION).astype("float32")
        nouvelles_meta = [{"titre": f"Nouveau {i}", "ville": "X"} for i in range(5)]

        nb_avant = index_cree.index.ntotal
        index_cree.ajouter_vecteurs(nouveaux, nouvelles_meta)

        assert index_cree.index.ntotal == nb_avant + 5

    def test_ajout_met_a_jour_metadata(self, index_cree: FAISSIndex):
        """
        Vérifie que les métadonnées en mémoire sont étendues après l'ajout.
        """
        np.random.seed(99)
        nouveaux = np.random.rand(5, DIMENSION).astype("float32")
        nouvelles_meta = [{"titre": f"Nouveau {i}", "ville": "X"} for i in range(5)]

        nb_avant = len(index_cree.metadata)
        index_cree.ajouter_vecteurs(nouveaux, nouvelles_meta)

        assert len(index_cree.metadata) == nb_avant + 5

    def test_ajout_sans_index_leve_value_error(self):
        """
        Vérifie qu'une ValueError est levée si l'index n'est pas initialisé.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        nouveaux = np.random.rand(3, DIMENSION).astype("float32")

        with pytest.raises(ValueError, match="non initialisé"):
            fi.ajouter_vecteurs(nouveaux, [{}] * 3)

    def test_ajout_dimension_incorrecte_leve_value_error(
        self, index_cree: FAISSIndex
    ):
        """
        Vérifie qu'une ValueError est levée si la dimension des nouveaux
        vecteurs ne correspond pas à celle de l'index.
        """
        mauvais = np.random.rand(3, DIMENSION + 5).astype("float32")

        with pytest.raises(ValueError, match="Dimension"):
            index_cree.ajouter_vecteurs(mauvais, [{}] * 3)

    def test_ajout_longueurs_incoherentes_leve_value_error(
        self, index_cree: FAISSIndex
    ):
        """
        Vérifie qu'une ValueError est levée si le nombre de vecteurs
        ne correspond pas au nombre de métadonnées.
        """
        nouveaux = np.random.rand(5, DIMENSION).astype("float32")

        with pytest.raises(ValueError, match="correspond pas"):
            index_cree.ajouter_vecteurs(nouveaux, [{}] * 3)

    def test_ajout_initialise_metadata_si_none(self, embeddings: np.ndarray):
        """
        Vérifie que les métadonnées sont initialisées si elles étaient None
        avant l'ajout.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexFlatL2")
        # metadata est None par défaut après creer_index

        nouveaux = np.random.rand(3, DIMENSION).astype("float32")
        nouvelles_meta = [{"titre": f"T{i}"} for i in range(3)]
        fi.ajouter_vecteurs(nouveaux, nouvelles_meta)

        assert fi.metadata is not None
        assert len(fi.metadata) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Reconstruction de l'index
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstruireIndex:
    """Tests de la méthode reconstruire_index."""

    def test_reconstruction_repart_de_zero(self, index_cree: FAISSIndex):
        """
        Vérifie que la reconstruction repart d'un index propre avec
        uniquement les nouveaux vecteurs.
        """
        np.random.seed(7)
        nouveaux_emb = np.random.rand(10, DIMENSION).astype("float32")
        nouvelles_meta = [{"titre": f"R{i}", "ville": "Y"} for i in range(10)]

        index_cree.reconstruire_index(nouveaux_emb, nouvelles_meta)

        assert index_cree.index.ntotal == 10

    def test_reconstruction_met_a_jour_metadata(self, index_cree: FAISSIndex):
        """
        Vérifie que les métadonnées sont remplacées après reconstruction.
        """
        np.random.seed(7)
        nouveaux_emb = np.random.rand(10, DIMENSION).astype("float32")
        nouvelles_meta = [{"titre": f"R{i}", "ville": "Y"} for i in range(10)]

        index_cree.reconstruire_index(nouveaux_emb, nouvelles_meta)

        assert len(index_cree.metadata) == 10
        assert index_cree.metadata[0]["titre"] == "R0"

    def test_reconstruction_efface_anciens_vecteurs(
        self, index_cree: FAISSIndex
    ):
        """
        Vérifie que les anciens vecteurs (NB_VECTEURS) ne sont plus présents
        après reconstruction avec un nombre différent de vecteurs.
        """
        np.random.seed(7)
        nouveaux_emb = np.random.rand(6, DIMENSION).astype("float32")
        nouvelles_meta = [{"titre": f"N{i}"} for i in range(6)]

        index_cree.reconstruire_index(nouveaux_emb, nouvelles_meta)

        # L'ancien index avait NB_VECTEURS vecteurs
        assert index_cree.index.ntotal != NB_VECTEURS
        assert index_cree.index.ntotal == 6


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Validation de cohérence
# ─────────────────────────────────────────────────────────────────────────────

class TestValiderCoherence:
    """Tests de la méthode valider_coherence."""

    def test_coherence_ok_retourne_true(self, index_cree: FAISSIndex):
        """
        Vérifie que la cohérence est validée quand index et métadonnées
        ont la même longueur.
        """
        assert index_cree.valider_coherence() is True

    def test_incoherence_retourne_false(self, index_cree: FAISSIndex):
        """
        Vérifie que la méthode retourne False quand les métadonnées
        ont moins d'entrées que l'index.
        """
        index_cree.metadata = index_cree.metadata[:5]

        assert index_cree.valider_coherence() is False

    def test_sans_index_leve_value_error(self):
        """
        Vérifie qu'une ValueError est levée si l'index n'est pas initialisé.
        """
        fi = FAISSIndex(dimension=DIMENSION)

        with pytest.raises(ValueError, match="non initialisé"):
            fi.valider_coherence()

    def test_sans_metadata_leve_value_error(self, embeddings: np.ndarray):
        """
        Vérifie qu'une ValueError est levée si les métadonnées ne sont pas
        chargées (None).
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexFlatL2")
        # fi.metadata est None par défaut

        with pytest.raises(ValueError, match="Métadonnées"):
            fi.valider_coherence()

    def test_metadata_trop_longue_retourne_false(
        self, index_cree: FAISSIndex
    ):
        """
        Vérifie que la méthode retourne False quand les métadonnées ont
        plus d'entrées que l'index.
        """
        index_cree.metadata = index_cree.metadata + [{"titre": "Extra"}]

        assert index_cree.valider_coherence() is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Recherche
# ─────────────────────────────────────────────────────────────────────────────

class TestRecherche:
    """Tests des méthodes rechercher et rechercher_avec_metadata."""

    def test_recherche_retourne_k_resultats(
        self, index_cree: FAISSIndex, embeddings: np.ndarray
    ):
        """
        Vérifie que la recherche retourne exactement k distances et indices.
        """
        query = embeddings[0]
        distances, indices = index_cree.rechercher(query, k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_premier_resultat_est_lui_meme(
        self, index_cree: FAISSIndex, embeddings: np.ndarray
    ):
        """
        Vérifie que le vecteur le plus proche de lui-même est à distance 0
        et correspond à l'indice 0.
        """
        query = embeddings[0]
        distances, indices = index_cree.rechercher(query, k=1)

        assert indices[0][0] == 0
        assert distances[0][0] == pytest.approx(0.0, abs=1e-5)

    def test_recherche_accepte_vecteur_1d(
        self, index_cree: FAISSIndex, embeddings: np.ndarray
    ):
        """
        Vérifie que la méthode accepte un vecteur de forme (dimension,)
        sans lever d'erreur (reshape interne attendu).
        """
        query = embeddings[0]  # Forme (DIMENSION,)
        assert query.ndim == 1

        distances, indices = index_cree.rechercher(query, k=2)

        assert distances.shape == (1, 2)

    def test_recherche_accepte_vecteur_2d(
        self, index_cree: FAISSIndex, embeddings: np.ndarray
    ):
        """
        Vérifie que la méthode accepte un vecteur de forme (1, dimension).
        """
        query = embeddings[0].reshape(1, -1)  # Forme (1, DIMENSION)
        distances, indices = index_cree.rechercher(query, k=2)

        assert distances.shape == (1, 2)

    def test_recherche_sans_index_leve_value_error(self):
        """
        Vérifie qu'une ValueError est levée si l'index n'est pas initialisé.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        query = np.random.rand(DIMENSION).astype("float32")

        with pytest.raises(ValueError, match="non initialisé"):
            fi.rechercher(query)

    def test_recherche_dimension_incorrecte_leve_value_error(
        self, index_cree: FAISSIndex
    ):
        """
        Vérifie qu'une ValueError est levée si la dimension du vecteur
        de requête ne correspond pas à celle de l'index.
        """
        mauvaise_query = np.random.rand(DIMENSION + 5).astype("float32")

        with pytest.raises(ValueError, match="Dimension"):
            index_cree.rechercher(mauvaise_query)

    def test_recherche_avec_metadata_retourne_liste(
        self, index_cree: FAISSIndex, embeddings: np.ndarray, metadata: list
    ):
        """
        Vérifie que rechercher_avec_metadata retourne une liste de
        dictionnaires.
        """
        query = embeddings[0]
        resultats = index_cree.rechercher_avec_metadata(query, metadata, k=3)

        assert isinstance(resultats, list)
        assert len(resultats) == 3

    def test_recherche_avec_metadata_champs_presents(
        self, index_cree: FAISSIndex, embeddings: np.ndarray, metadata: list
    ):
        """
        Vérifie que chaque résultat contient les champs rank, distance,
        score et les champs de métadonnées.
        """
        query = embeddings[0]
        resultats = index_cree.rechercher_avec_metadata(query, metadata, k=3)

        for res in resultats:
            assert "rank" in res
            assert "distance" in res
            assert "score" in res
            assert "titre" in res
            assert "ville" in res

    def test_score_entre_0_et_1(
        self, index_cree: FAISSIndex, embeddings: np.ndarray, metadata: list
    ):
        """
        Vérifie que les scores de similarité sont compris dans [0, 1].
        """
        query = embeddings[0]
        resultats = index_cree.rechercher_avec_metadata(query, metadata, k=5)

        for res in resultats:
            assert 0.0 <= res["score"] <= 1.0

    def test_rang_croissant(
        self, index_cree: FAISSIndex, embeddings: np.ndarray, metadata: list
    ):
        """
        Vérifie que les rangs sont bien croissants (1, 2, 3, ...).
        """
        query = embeddings[0]
        resultats = index_cree.rechercher_avec_metadata(query, metadata, k=5)

        rangs = [res["rank"] for res in resultats]
        assert rangs == list(range(1, len(rangs) + 1))

    def test_distance_croissante(
        self, index_cree: FAISSIndex, embeddings: np.ndarray, metadata: list
    ):
        """
        Vérifie que les distances sont triées par ordre croissant
        (du plus proche au plus éloigné).
        """
        query = embeddings[0]
        resultats = index_cree.rechercher_avec_metadata(query, metadata, k=5)

        distances = [res["distance"] for res in resultats]
        assert distances == sorted(distances)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Sauvegarde et chargement
# ─────────────────────────────────────────────────────────────────────────────

class TestSauvegardeChargement:
    """Tests des méthodes sauvegarder et charger."""

    def test_sauvegarde_cree_fichier_index(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que la sauvegarde crée le fichier faiss.index.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            assert (Path(tmpdir) / "faiss.index").exists()

    def test_sauvegarde_cree_fichier_metadata(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que la sauvegarde crée le fichier metadata.json.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            assert (Path(tmpdir) / "metadata.json").exists()

    def test_sauvegarde_cree_fichier_stats(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que la sauvegarde crée le fichier stats_index.json.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            assert (Path(tmpdir) / "stats_index.json").exists()

    def test_sauvegarde_sans_index_leve_value_error(self):
        """
        Vérifie qu'une ValueError est levée si l'index n'est pas initialisé.
        """
        fi = FAISSIndex(dimension=DIMENSION)

        with pytest.raises(ValueError, match="non initialisé"):
            fi.sauvegarder("/tmp/test_faiss")

    def test_chargement_restaure_ntotal(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que le chargement restaure le bon nombre de vecteurs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            fi2 = FAISSIndex(dimension=DIMENSION)
            fi2.charger(tmpdir)

            assert fi2.index.ntotal == NB_VECTEURS

    def test_chargement_restaure_metadata(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que le chargement restaure les métadonnées avec la bonne
        longueur.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            fi2 = FAISSIndex(dimension=DIMENSION)
            fi2.charger(tmpdir)

            assert len(fi2.metadata) == NB_VECTEURS

    def test_chargement_coherence_apres_restauration(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que l'index rechargé est cohérent avec ses métadonnées.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            fi2 = FAISSIndex(dimension=DIMENSION)
            fi2.charger(tmpdir)

            assert fi2.valider_coherence() is True

    def test_chargement_fichier_absent_leve_file_not_found(self):
        """
        Vérifie qu'une FileNotFoundError est levée si le fichier faiss.index
        est absent du dossier spécifié.
        """
        fi = FAISSIndex(dimension=DIMENSION)

        with pytest.raises(FileNotFoundError):
            fi.charger("/chemin/inexistant")

    def test_sauvegarde_utilise_metadata_en_memoire(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que si metadata=None est passé à sauvegarder, les métadonnées
        en mémoire (self.metadata) sont utilisées.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # self.metadata est déjà défini dans la fixture index_cree
            index_cree.sauvegarder(tmpdir, metadata=None)

            assert (Path(tmpdir) / "metadata.json").exists()

            with open(Path(tmpdir) / "metadata.json", "r") as f:
                meta_chargee = json.load(f)

            assert len(meta_chargee) == NB_VECTEURS

    def test_stats_json_contient_champs_attendus(
        self, index_cree: FAISSIndex, metadata: list
    ):
        """
        Vérifie que le fichier stats_index.json contient les champs
        dimension, nombre_vecteurs et type_index.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_cree.sauvegarder(tmpdir, metadata)

            with open(Path(tmpdir) / "stats_index.json", "r") as f:
                stats = json.load(f)

        assert "dimension" in stats
        assert "nombre_vecteurs" in stats
        assert "type_index" in stats
        assert stats["dimension"] == DIMENSION
        assert stats["nombre_vecteurs"] == NB_VECTEURS


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Statistiques
# ─────────────────────────────────────────────────────────────────────────────

class TestStatistiques:
    """Tests de la méthode obtenir_statistiques."""

    def test_champs_presents(self, index_cree: FAISSIndex):
        """
        Vérifie que les statistiques contiennent tous les champs attendus.
        """
        stats = index_cree.obtenir_statistiques()

        for champ in [
            "dimension",
            "nombre_vecteurs",
            "type_index",
            "is_trained",
            "nb_metadata",
            "coherent",
        ]:
            assert champ in stats, f"Champ manquant : {champ}"

    def test_valeurs_correctes(self, index_cree: FAISSIndex):
        """
        Vérifie que les valeurs des statistiques sont cohérentes avec
        l'état de l'index.
        """
        stats = index_cree.obtenir_statistiques()

        assert stats["dimension"] == DIMENSION
        assert stats["nombre_vecteurs"] == NB_VECTEURS
        assert stats["nb_metadata"] == NB_VECTEURS
        assert stats["coherent"] is True

    def test_coherent_false_si_incoherence(self, index_cree: FAISSIndex):
        """
        Vérifie que coherent vaut False dans les statistiques si les
        métadonnées ne correspondent pas à l'index.
        """
        index_cree.metadata = index_cree.metadata[:3]
        stats = index_cree.obtenir_statistiques()

        assert stats["coherent"] is False

    def test_index_non_initialise_retourne_dict_vide(self):
        """
        Vérifie que les statistiques retournent un dictionnaire vide si
        l'index n'est pas initialisé.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        stats = fi.obtenir_statistiques()

        assert stats == {}

    def test_nb_metadata_none_si_pas_de_metadata(
        self, embeddings: np.ndarray
    ):
        """
        Vérifie que nb_metadata vaut None dans les statistiques si aucune
        métadonnée n'est chargée en mémoire.
        """
        fi = FAISSIndex(dimension=DIMENSION)
        fi.creer_index(embeddings, index_type="IndexFlatL2")
        # fi.metadata est None par défaut

        stats = fi.obtenir_statistiques()

        assert stats["nb_metadata"] is None
        assert stats["coherent"] is None
