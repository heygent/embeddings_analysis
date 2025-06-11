import pytest
from pathlib import Path
import pandas as pd
import shutil
from embeddings_analysis.loader import CachedEmbeddingsLoader, TransformersEmbeddingsLoader
from embeddings_analysis.dataclasses import RandomEmbeddingsMeta, RangeEmbeddingsMeta

@pytest.fixture
def test_model_id():
    return "bert-base-uncased"

@pytest.fixture
def cache_dir(tmp_path):
    cache_path = tmp_path / "saved_embeddings"
    cache_path.mkdir()
    yield cache_path
    if cache_path.exists():
        shutil.rmtree(cache_path)

@pytest.fixture
def loader(test_model_id, cache_dir):
    return CachedEmbeddingsLoader(
        model_id=test_model_id,
        use_transformers_fallback=True,
        data_path=cache_dir
    )

def test_range_embeddings_caching(loader, cache_dir):
    # Generate and cache embeddings
    data1 = loader.range(stop=5)
    assert isinstance(data1.data, pd.DataFrame)
    assert len(data1.data) == 5
    
    cache_file = loader.path_for_metadata(RangeEmbeddingsMeta(loader.model_id, 5))
    assert cache_file.exists()

    # Create new loader and verify cache loading
    read_loader = CachedEmbeddingsLoader(
        model_id=loader.model_id,
        use_transformers_fallback=False,
        data_path=cache_dir
    )
    data2 = read_loader.range(stop=5)
    pd.testing.assert_frame_equal(data1.data, data2.data)

def test_random_embeddings_caching(loader, cache_dir):
    # Generate and cache embeddings
    data1 = loader.random(n=10, seed=42)
    assert isinstance(data1.data, pd.DataFrame)
    assert len(data1.data) == 10
    
    cache_file = loader.path_for_metadata(RandomEmbeddingsMeta(loader.model_id, 10, 42))
    assert cache_file.exists()

    # Create new loader and verify cache loading
    read_loader = CachedEmbeddingsLoader(
        model_id=loader.model_id,
        use_transformers_fallback=False,
        data_path=cache_dir
    )
    data2 = read_loader.random(n=10, seed=42)
    pd.testing.assert_frame_equal(data1.data, data2.data)

def test_cache_not_found(test_model_id, cache_dir):
    read_loader = CachedEmbeddingsLoader(
        model_id=test_model_id,
        use_transformers_fallback=False,
        data_path=cache_dir
    )
    with pytest.raises(FileNotFoundError):
        read_loader.range(stop=5)
    with pytest.raises(FileNotFoundError):
        read_loader.random(n=10, seed=42)

def test_base_path_creation(test_model_id, tmp_path):
    new_cache_dir = tmp_path / "new_cache"
    loader = CachedEmbeddingsLoader(
        model_id=test_model_id,
        use_transformers_fallback=True,
        data_path=new_cache_dir
    )
    
    data = loader.range(stop=5)
    assert new_cache_dir.exists()
    assert len(list(new_cache_dir.glob("*.parquet"))) == 1
    assert isinstance(data.data, pd.DataFrame)
