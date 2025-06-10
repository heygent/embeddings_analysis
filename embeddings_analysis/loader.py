from __future__ import annotations

from functools import cache, cached_property
from pathlib import Path

import marimo as mo
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from embeddings_analysis.dataclasses import (
    RandomEmbeddingsMeta,
    EmbeddingsData,
    RangeEmbeddingsMeta
)


class TransformersEmbeddingsLoader:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    @cached_property
    def embeddings(self):
        model = AutoModel.from_pretrained(self.model_id)
        model.eval()
        embeddings = model.embed_tokens
        del model
        return embeddings

    def __repr__(self):
        return f'EmbeddingsLoader("{self.model_id}")'
    
    def _tokenize(self, words):
        return self.tokenizer(
            words,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]

    def _df_from_token_ids(self, token_ids: np.ndarray, tokens: list[str] | None) -> EmbeddingsData:
        with torch.no_grad():
            token_ids = torch.from_numpy(token_ids)
            extracted_embeddings = self.embeddings.forward(token_ids).squeeze().numpy()
        
        if tokens is None:
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

        return pd.DataFrame(extracted_embeddings, index=pd.Index(tokens, name="Token"))

    def _df_from_tokens(self, token_list: list[str]) -> EmbeddingsData:
        token_ids = self._tokenize(token_list)
        return self._df_from_token_ids(token_ids, token_list)
    
    @cache
    def smallest_multitoken_number(self, upper_limit=1200):
        for num in range(upper_limit):
            tokens = self.tokenizer.tokenize(str(num))
            if len(tokens) > 1:
                return num
        raise ValueError(f"All numbers from 0 to {upper_limit=} are single token. ")

    def range(self, stop=None) -> EmbeddingsData:
        if stop is None:
            stop = self.smallest_multitoken_number()
        
        return EmbeddingsData(
            RangeEmbeddingsMeta(
                self.model_id,
                stop
            ),
            self._df_from_tokens([str(i) for i in range(stop)]),
        )

    def random(self, n=1000, seed=1234) -> EmbeddingsData:
        generator = np.random.default_rng(seed)
        
        random_token_ids = generator.integers(
            low=0,
            high=self.embeddings.num_embeddings,
            size=n,
        )

        return EmbeddingsData(
            RandomEmbeddingsMeta(
                self.model_id,
                n,
                seed
            ),
            self._df_from_token_ids(random_token_ids, None),
        )

class CachedEmbeddingsLoader:
    def __init__(self, model_id: str, fallback: bool = False):
        self.model_id = model_id
        self.fallback = fallback
    
    @cached_property
    def transformers_loader(self) -> TransformersEmbeddingsLoader:
        return TransformersEmbeddingsLoader(self.model_id)
    
    @staticmethod
    def path_for_metadata(meta):
        if mo.running_in_notebook():
            base_path = mo.notebook_location()
        else:
            base_path = Path(__file__).parent.parent
    
        return base_path / "saved_embeddings" / f"{meta.id}.parquet"
    
    def range(self, stop=1000):
        meta = RangeEmbeddingsMeta(self.model_id, stop)
        path = self.path_for_metadata(meta)

        if path.exists():
            data = pd.read_parquet(path)
        
        if self.fallback:
            data = self.transformers_loader.range(stop)
            data.data.to_parquet(path)
            return data
        
        raise FileNotFoundError(f"Cached embeddings for {meta} not found at {path}.")
        
    def random(self, n=1000, seed=1234) -> EmbeddingsData:
        meta = RandomEmbeddingsMeta(self.model_id, n, seed)
        path = self.path_for_metadata(meta)

        if path.exists():
            data = pd.read_parquet(path)
        
        if self.fallback:
            data = self.transformers_loader.random(n, seed)
            data.data.to_parquet(path)
            return data
        
        raise FileNotFoundError(f"Cached embeddings for {meta} not found at {path}.")