from __future__ import annotations

from functools import cache

import torch
from transformers import AutoModel, AutoTokenizer

from embeddings_analysis.dataclasses import EmbeddingsData


def load_embeddings(model_id):
    """Load the embeddings from a pretrained model and discards the rest of the model."""
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    embeddings = model.embed_tokens
    del model
    return embeddings


class EmbeddingsLoader:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.embeddings = load_embeddings(model_id)

    def __repr__(self):
        return f'EmbeddingsLoader("{self.model_id}")'

    def tokenize(self, words):
        return self.tokenizer(
            words,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]

    def from_tokens(self, tokens, label) -> EmbeddingsData:
        with torch.no_grad():
            extracted_embeddings = self.embeddings.forward(tokens).squeeze().numpy()

        return EmbeddingsData(self.model_id, label, extracted_embeddings)

    def from_words(self, words, label) -> EmbeddingsData:
        return self.from_tokens(self.tokenize(words), label)
    
    @cache
    def smallest_multitoken_number(self, upper_limit=1200):
        for num in range(upper_limit):
            tokens = self.tokenizer.tokenize(str(num))
            if len(tokens) > 1:
                return num
        raise ValueError(f"All numbers from 0 to {upper_limit} are single token. ")


    def numbers(self, upper_limit=None) -> EmbeddingsData:
        if upper_limit is None:
            upper_limit = self.smallest_multitoken_number()

        return self.from_words(
            [str(i) for i in range(upper_limit)],
            f"Number embeddings between 0-{upper_limit - 1}",
        )

    def random(self, n=1000, seed=None) -> EmbeddingsData:
        if seed is not None:
            torch.manual_seed(seed)
        random_tokens = torch.randint(0, self.embeddings.num_embeddings, (n,))

        return self.from_tokens(random_tokens, f"Random embeddings {n=}")
