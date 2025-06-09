from __future__ import annotations

from functools import cache, cached_property
from pathlib import Path

import marimo as mo
import torch
from transformers import AutoModel, AutoTokenizer

from embeddings_analysis.dataclasses import (
    RangeEmbeddingsMeta,
    RandomEmbeddingsMeta,
    EmbeddingsData
)


def get_cached_embeddings_path(id: str) -> Path:
    if mo.running_in_notebook():
        base_path = mo.notebook_location()
    else:
        base_path = Path(__file__).parent
    
    return base_path / "saved_embeddings" / f"{id}.pt"


def load_from_model(model_id):
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    embeddings = model.embed_tokens
    del model
    return embeddings


class EmbeddingsLoader:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    @cached_property
    def embeddings(self):
        return load_from_model(self.model_id)

    def __repr__(self):
        return f'EmbeddingsLoader("{self.model_id}")'

    def _tokenize(self, words):
        return self.tokenizer(
            words,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]

    def _from_tokens(self, tokens, meta) -> EmbeddingsData:
        with torch.no_grad():
            extracted_embeddings = self.embeddings.forward(tokens).squeeze().numpy()

        return EmbeddingsData(meta, extracted_embeddings)

    def _from_words(self, words, meta) -> EmbeddingsData:
        return self._from_tokens(self._tokenize(words), meta)
    
    @cache
    def smallest_multitoken_number(self, upper_limit=1200):
        for num in range(upper_limit):
            tokens = self.tokenizer.tokenize(str(num))
            if len(tokens) > 1:
                return num
        raise ValueError(f"All numbers from 0 to {upper_limit} are single token. ")


    def range(self, stop=None) -> EmbeddingsData:
        if stop is None:
            stop = self.smallest_multitoken_number()

        return self._from_words(
            [str(i) for i in range(stop)],
            RangeEmbeddingsMeta(self.model_id, stop)
        )

    def random(self, n=1000, seed=None) -> EmbeddingsData:
        if seed is not None:
            torch.manual_seed(seed)
        
        random_tokens_ids = torch.randint(0, self.embeddings.num_embeddings, (n,))

        token_list = self.tokenizer.convert_ids_to_tokens(random_tokens_ids.tolist())
        meta = RandomEmbeddingsMeta(self.model_id, n, seed, token_list)

        return self._from_tokens(random_tokens_ids, meta)
