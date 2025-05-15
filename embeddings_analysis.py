from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import warnings

import altair as alt
import pandas as pd
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


@cache
def smallest_multitoken_number(model_id, upper_limit=1200):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for num in range(upper_limit):
        tokens = tokenizer.tokenize(str(num))
        if len(tokens) > 1:
            return num


def load_embeddings(model_id):
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

    def numbers(self, upper_limit=None) -> EmbeddingsData:
        if upper_limit is None:
            upper_limit = smallest_multitoken_number(self.model_id)

        return self.from_words(
            [str(i) for i in range(upper_limit)],
            f"Number embeddings between 0-{upper_limit - 1}",
        )

    def random(self, n=1000, seed=None) -> EmbeddingsData:
        if seed is not None:
            torch.manual_seed(seed)
        random_tokens = torch.randint(0, self.embeddings.num_embeddings, (n,))

        return self.from_tokens(random_tokens, f"Random embeddings {n=}")


@dataclass
class EmbeddingsData:
    model_id: str
    label: str
    data: np.ndarray

    def __str__(self):
        return f"({self.model_id}) {self.label}"

    def dim_reduction(self, obj):
        return EmbeddingsDimReduction(self, obj, obj.fit_transform(self.data))


default_props = {"width": 600, "height": 500}


@dataclass
class EmbeddingsDimReduction:
    embeddings_data: EmbeddingsData
    reduction: object
    transformed: np.ndarray

    def plot(
        self, x_component: int = 0, y_component: int = 1, colorscheme: str = "viridis"
    ) -> alt.Chart:
        """Create a 2D scatter plot of two principal components."""
        projection_df = pd.DataFrame(
            {
                f"Component{x_component + 1}": self.transformed[:, x_component],
                f"Component{y_component + 1}": self.transformed[:, y_component],
                "Number": np.arange(self.transformed.shape[0]),
            }
        )

        return (
            alt.Chart(projection_df)
            .mark_circle(size=60)
            .encode(
                x=alt.X(
                    f"Component{x_component + 1}:Q",
                    title=f"Component {x_component + 1}",
                ),
                y=alt.Y(
                    f"Component{y_component + 1}:Q",
                    title=f"Component {y_component + 1}",
                ),
                color=alt.Color(
                    "Number:Q",
                    scale=alt.Scale(scheme=colorscheme),
                    title="Number Value",
                ),
                tooltip=[
                    "Number",
                    f"Component{x_component + 1}",
                    f"Component{y_component + 1}",
                ],
            )
            .properties(
                title=f"{self.embeddings_data}: {self.reduction.__class__.__name__}",
                **default_props,
            )
            .interactive()
        )


def dim_reductions(embeddings_data: EmbeddingsData, random_state=None):
    pca = PCA(n_components=100)

    with warnings.catch_warnings(category=UserWarning):
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            n_iter=1000,
        )

    umap = UMAP(
        n_components=2,
        random_state=random_state
    )

    return [embeddings_data.dim_reduction(reduction) for reduction in (pca, tsne, umap)]


def plot_compared_dim_reductions(
    reductions_a: list[EmbeddingsDimReduction],
    reductions_b: list[EmbeddingsDimReduction],
):
    return alt.vconcat(
        *[
            alt.hconcat(r_a.plot(), r_b.plot())
            for r_a, r_b in zip(reductions_a, reductions_b)
        ]
    )


def plot_embeddings(model_id: str):
    loader = EmbeddingsLoader(model_id)
    numbers_data = loader.numbers()
    random_data = loader.random()
    reductions_a = dim_reductions(numbers_data)
    reductions_b = dim_reductions(random_data)

    return plot_compared_dim_reductions(reductions_a, reductions_b)
