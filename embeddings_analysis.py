from dataclasses import dataclass
from functools import cache, cached_property
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
def smallest_multitoken_number(model_id, lower_limit=0, upper_limit=1200):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for num in range(lower_limit, upper_limit):
        tokens = tokenizer.tokenize(str(num))
        if len(tokens) > 1:
            return num


def load_embeddings(model_id):
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    embeddings = model.embed_tokens
    del model
    return embeddings

@dataclass
class EmbeddingsData:
    model_id: str
    label: str
    data: np.ndarray

    def __str__(self):
        return f'({self.model_id}) {self.label}'
    
    @cached_property
    def pca(self):
        return PCA(n_components=100)
    
    @cached_property
    def pca_result(self):
        return self.pca.fit_transform(self.data)

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
            upper_limit = smallest_multitoken_number(self.tokenizer.name_or_path)

        return self.from_words(
            [str(i) for i in range(upper_limit)],
            f'Number embeddings between 0-{upper_limit-1}',
        )
    
    def random(self, n=1000, seed=None) -> EmbeddingsData:
        if seed is not None:
            torch.manual_seed(seed)
        random_tokens = torch.randint(0, self.embeddings.num_embeddings, (n,))

        return self.from_tokens(random_tokens, f'Random embeddings {n=}')

default_props = {
    'width': 400,
}

    
def plot_pca(self: EmbeddingsData, colorscheme='viridis'):
    pca_df = pd.DataFrame(self.pca_result[:, 0:2], columns=['PC1', 'PC2'])
    pca_df['number'] = range(1000)

    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x='PC1:Q',
        y='PC2:Q',
        color=alt.Color('number:Q', scale=alt.Scale(scheme=colorscheme)),
        tooltip=['number:Q', 'PC1:Q', 'PC2:Q']
    ).properties(
        title=f'{self}: PCA',
        **default_props
    ).interactive()

    return chart

def plot_tsne(embeddings_data: EmbeddingsData, n_iter=1000, random_state=None, colorscheme='viridis'):
    raw_data = embeddings_data.data

    with warnings.catch_warnings(category=UserWarning):
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            n_iter=n_iter,
        )
        tsne_result = tsne.fit_transform(raw_data)

    tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
    tsne_df['number'] = range(1000)

    chart = alt.Chart(tsne_df).mark_circle(size=60).encode(
        x='t-SNE1:Q',
        y='t-SNE2:Q',
        color=alt.Color('number:Q', scale=alt.Scale(scheme=colorscheme)),
        tooltip=['number:Q', 't-SNE1:Q', 't-SNE2:Q']
    ).properties(
        title=f'{embeddings_data}: t-SNE',
        **default_props
    ).interactive()

    return chart

def plot_umap(embeddings_data: EmbeddingsData, n_epochs=2000, random_state=None, colorscheme='viridis'):
    # setting random_state will disable parallelization

    raw_data = embeddings_data.data

    with warnings.catch_warnings(category=UserWarning):
        reducer = UMAP(
            n_components=2,
            random_state=random_state,
            n_epochs=n_epochs,
        )
        umap_result = reducer.fit_transform(raw_data)

    umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    umap_df['number'] = range(1000)

    chart = alt.Chart(umap_df).mark_circle(size=60).encode(
        x='UMAP1:Q',
        y='UMAP2:Q',
        color=alt.Color('number:Q', scale=alt.Scale(scheme=colorscheme)),
        tooltip=['number:Q', 'UMAP1:Q', 'UMAP2:Q']
    ).properties(
        title=f'{embeddings_data}: UMAP',
        **default_props
    ).interactive()

    return chart

def plot_embeddings(model_id):
    loader = EmbeddingsLoader(model_id)
    number_data = loader.numbers()
    random_data = loader.random()
    del loader

    return alt.vconcat(
        alt.hconcat(plot_pca(number_data), plot_pca(random_data)),
        alt.hconcat(plot_tsne(number_data), plot_tsne(random_data)),
        alt.hconcat(plot_umap(number_data), plot_umap(random_data))
    )