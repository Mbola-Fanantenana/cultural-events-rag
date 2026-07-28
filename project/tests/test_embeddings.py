import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.embeddings.mistral_embeddings import MistralEmbedder


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mock_settings(monkeypatch):
    """Mock des variables de config.settings"""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    with patch("config.settings.MISTRAL_API_KEY", "fake_key"), \
         patch("config.settings.MISTRAL_EMBEDDING_MODEL", "test-model"), \
         patch("config.settings.PROXIES", {}), \
         patch("config.settings.EMBEDDING_BATCH_SIZE", 2), \
         patch("config.settings.EMBEDDING_DELAY", 0):
        yield


@pytest.fixture
def embedder(mock_settings):
    """Instance de MistralEmbedder avec client mocké"""
    emb = MistralEmbedder(api_key="fake_key", use_proxy=False)

    emb.client = MagicMock()
    return emb


def fake_response(vectors, tokens=10):
    """Fabrique une réponse fake de l'API"""
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    response.usage.total_tokens = tokens
    return response


# =========================================================
# TESTS : INITIALISATION
# =========================================================

def test_init_ok(mock_settings):
    emb = MistralEmbedder(api_key="fake_key", use_proxy=False)
    assert emb.api_key == "fake_key"
    assert emb.dimension is None


def test_init_no_api_key(monkeypatch):
    with patch("config.settings.MISTRAL_API_KEY", None):
        with pytest.raises(ValueError):
            MistralEmbedder(api_key=None)


# =========================================================
# TESTS : EMBEDDINGS (SUCCESS)
# =========================================================

def test_creer_embeddings_success(embedder):
    embedder.client.embeddings.create.return_value = fake_response(
        [[1.0, 2.0], [3.0, 4.0]]
    )

    texts = ["hello", "world"]
    embeddings, errors = embedder.creer_embeddings(texts, batch_size=2)

    assert embeddings.shape == (2, 2)
    assert len(errors) == 0
    assert embedder.dimension == 2


def test_creer_embeddings_empty():
    emb = MistralEmbedder(api_key="fake_key", use_proxy=False)
    arr, errors = emb.creer_embeddings([])
    assert arr.shape == (0, 0)
    assert errors == []


# =========================================================
# TESTS : RETRY + ERREURS
# =========================================================

def test_retry_success(embedder):
    # 1er appel échoue, 2e réussit
    embedder.client.embeddings.create.side_effect = [
        Exception("fail"),
        fake_response([[1.0, 2.0]])
    ]

    embeddings, errors = embedder.creer_embeddings(["test"], batch_size=1)

    assert embeddings.shape == (1, 2)
    assert errors == []


def test_retry_failure(embedder):
    embedder.client.embeddings.create.side_effect = Exception("fail")

    with pytest.raises(RuntimeError):
        embedder.creer_embeddings(["test"], batch_size=1)


def test_all_batches_fail(embedder):
    embedder.client.embeddings.create.side_effect = Exception("fail")

    with pytest.raises(RuntimeError):
        embedder.creer_embeddings(["a", "b"], batch_size=1)


# =========================================================
# TESTS : SANITIZATION
# =========================================================

def test_sanitize_inputs(embedder):
    embedder.client.embeddings.create.return_value = fake_response(
        [[1.0, 2.0], [3.0, 4.0]]
    )

    texts = ["ok", None]
    embeddings, _ = embedder.creer_embeddings(texts, batch_size=2)

    assert embeddings.shape == (2, 2)


# =========================================================
# TESTS : DATAFRAME
# =========================================================

def test_dataframe_embeddings(embedder):
    embedder.client.embeddings.create.return_value = fake_response(
        [[1.0, 2.0], [3.0, 4.0]]
    )

    df = pd.DataFrame({
        "texte_complet": ["a", "b"],
        "id": ["1", "2"],
        "titre": ["t1", "t2"],
    })

    emb, metadata, errors = embedder.creer_embeddings_depuis_dataframe(df)

    assert emb.shape == (2, 2)
    assert len(metadata) == 2
    assert errors == []


def test_dataframe_missing_column(embedder):
    df = pd.DataFrame({"wrong": ["a"]})

    with pytest.raises(KeyError):
        embedder.creer_embeddings_depuis_dataframe(df)


# =========================================================
# TESTS : VALEUR PAR DEFAUT
# =========================================================

def test_valeur_ou_defaut():
    row = pd.Series({"a": None})

    val = MistralEmbedder._valeur_ou_defaut(row, "a", "x")
    assert val == "x"

    val = MistralEmbedder._valeur_ou_defaut(row, "b", "y")
    assert val == "y"


# =========================================================
# TESTS : SAVE / LOAD
# =========================================================

def test_save_and_load(tmp_path, embedder):
    embeddings = np.array([[1.0, 2.0]])
    metadata = [{"id": "1"}]

    embedder.sauvegarder_embeddings(
        embeddings,
        metadata,
        output_dir=tmp_path
    )

    emb2, meta2 = embedder.charger_embeddings(tmp_path)

    assert np.array_equal(embeddings, emb2)
    assert metadata == meta2


def test_save_inconsistent_length(embedder, tmp_path):
    embeddings = np.array([[1.0, 2.0]])
    metadata = []

    with pytest.raises(ValueError):
        embedder.sauvegarder_embeddings(embeddings, metadata, tmp_path)


def test_load_missing_files(embedder, tmp_path):
    with pytest.raises(FileNotFoundError):
        embedder.charger_embeddings(tmp_path)


# =========================================================
# TESTS : STATS
# =========================================================

def test_stats(embedder):
    stats = embedder.obtenir_statistiques()
    assert isinstance(stats, dict)
    assert "total_events" in stats