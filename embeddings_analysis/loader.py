from abc import ABC, abstractmethod
from functools import cache, cached_property
from os import PathLike
from pathlib import Path

import marimo as mo
import numpy as np
import pandas as pd
import torch

from transformers import AutoModel, AutoTokenizer

from embeddings_analysis.dataclasses import (
    EmbeddingsData,
    RandomEmbeddingsMeta,
    RangeEmbeddingsMeta
)

class EmbeddingsLoader(ABC):
    @abstractmethod
    def range(self, stop: int) -> EmbeddingsData[RangeEmbeddingsMeta]:
        pass

    @abstractmethod
    def random(self, n: int, seed: int) -> EmbeddingsData[RandomEmbeddingsMeta]:
        pass


class TransformersEmbeddingsLoader(EmbeddingsLoader):
    def __init__(self, model_id):
        self.model_id = model_id
    
    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id)
    
    @cached_property
    def embeddings(self):
        model = AutoModel.from_pretrained(self.model_id)
        model.eval()
        embeddings = model.embed_tokens
        del model
        return embeddings
    
    def __repr__(self):
        return f'EmbeddingsLoader("{self.model_id}")'

    def tokenize(self, words):
        return self.tokenizer(
            words,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]

    def embeddings_for_token_ids(self, token_ids: np.ndarray):
        with torch.no_grad():
            token_ids = torch.tensor(token_ids)
            return self.embeddings.forward(token_ids).squeeze().numpy()
    
    def embeddings_dataframe(self, token_ids: np.ndarray | None = None, tokens: list[str] | None = None) -> pd.DataFrame:
        if token_ids is None and tokens is None:
            raise ValueError('Either tokens or token_ids has to be specified')
        elif token_ids is None:        
            token_ids = self.tokenize(tokens)
        elif tokens is None:
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        
        embeddings_np = self.embeddings_for_token_ids(token_ids)
        
        df = pd.DataFrame(embeddings_np)
        df.columns = df.columns.astype(str)
        df.insert(0, "Token ID", token_ids)
        df.insert(0, "Token", tokens)
        return df
    
    @cache
    def smallest_multitoken_number(self, upper_limit=1200):
        for num in range(upper_limit):
            tokens = self.tokenizer.tokenize(str(num))
            if len(tokens) > 1:
                return num
        raise ValueError(f"All numbers from 0 to {upper_limit=} are single token. ")
    
    def range(self, stop=None):
        if stop is None:
            stop = self.smallest_multitoken_number()
        
        meta = RangeEmbeddingsMeta(
            self.model_id,
            stop
        )
        tokens = list(str(n) for n in range(stop))
        token_ids = self.tokenize(tokens)

        return EmbeddingsData(meta, self.embeddings_dataframe(token_ids=token_ids))

    def random(self, n=1000, seed=1234):
        meta = RandomEmbeddingsMeta(
            self.model_id,
            seed,
            n,
        ),

        generator = np.random.default_rng(seed)
        
        random_token_ids = generator.integers(
            low=0,
            high=self.embeddings.num_embeddings,
            size=n,
        )

        return EmbeddingsData(meta, self.embeddings_dataframe(token_ids=random_token_ids))

class CachedEmbeddingsLoader(EmbeddingsLoader):
    def __init__(self, model_id: str, use_transformers_fallback: bool = False, data_path: PathLike = None):
        self.model_id = model_id
        self.use_transformers_fallback = use_transformers_fallback

        if data_path is None:
            if mo.running_in_notebook():
                data_path = mo.notebook_location() / "saved_embeddings"
            else:
                data_path = Path(__file__).parent.parent / "saved_embeddings"
                data_path.mkdir(parents=True, exist_ok=True)
    
        self.data_path = Path(data_path)

    @cached_property
    def transformers_loader(self) -> TransformersEmbeddingsLoader:
        return TransformersEmbeddingsLoader(self.model_id)
    
    def path_for_metadata(self, meta):
        return self.data_path / f"{meta.id}.parquet"
    
    def range(self, stop=1000):
        meta = RangeEmbeddingsMeta(self.model_id, stop)
        path = self.path_for_metadata(meta)

        if path.exists():
            df = pd.read_parquet(path)
            return EmbeddingsData(meta, df)
        
        if self.use_transformers_fallback:
            data = self.transformers_loader.range(stop)
            data.embeddings.to_parquet(path)
            return data
        
        raise FileNotFoundError(f"Cached embeddings for {meta} not found at {path}.")

    def random(self, n=1000, seed=1234):
        meta = RandomEmbeddingsMeta(self.model_id, n, seed)
        path = self.path_for_metadata(meta)

        if path.exists():
            df = pd.read_parquet(path)
            return EmbeddingsData(meta, df)
        
        if self.use_transformers_fallback:
            data = self.transformers_loader.random(n, seed)
            data.embeddings.to_parquet(path)
            return data
        
        raise FileNotFoundError(f"Cached embeddings for {meta} not found at {path}.")


def get_loader(model_id: str, use_transformers_fallback = True) -> EmbeddingsLoader:
    return CachedEmbeddingsLoader(model_id, use_transformers_fallback)