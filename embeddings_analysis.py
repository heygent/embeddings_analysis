from functools import cache
import warnings

import altair as alt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch

from transformers import AutoModel, AutoTokenizer

import umap
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.embeddings = load_embeddings(model_id)
    
    def __repr__(self):
        return f'EmbeddingsLoader("{self.tokenizer.name_or_path}")'
    
    def tokenize(self, words):
        return self.tokenizer(
            words,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]
    
    def from_tokens(self, tokens):
        with torch.no_grad():
            extracted_embeddings = self.embeddings.forward(tokens).squeeze()
        return extracted_embeddings

    def from_words(self, words):
        return self.from_tokens(self.tokenize(words))
    
    def numbers(self, upper_limit=None):
        if upper_limit is None:
            upper_limit = smallest_multitoken_number(self.tokenizer.name_or_path)

        return self.from_words([str(i) for i in range(upper_limit)])
    
    def random(self, n=1000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        random_tokens = torch.randint(0, self.embeddings.num_embeddings, (n,))
        return self.from_tokens(random_tokens)


class EmbeddingsViz:
    def __init__(self, embeddings, model_id, label, color_scheme='viridis', **alt_props):
        self.embeddings = embeddings
        self.model_id = model_id
        self.label = label

        self.color_scheme = color_scheme
        alt_props.setdefault('width', 500)
        self.alt_props = alt_props
    
    def __str__(self):
        return f'({self.model_id}) {self.label}'

    def pca(self):
        embeddings_array = self.embeddings

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings_array)

        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        pca_df['number'] = range(1000)

        chart = alt.Chart(pca_df).mark_circle(size=60).encode(
            x='PC1:Q',
            y='PC2:Q',
            color=alt.Color('number:Q', scale=alt.Scale(scheme=self.color_scheme)),
            tooltip=['number:Q', 'PC1:Q', 'PC2:Q']
        ).properties(
            title=f'{self}: PCA',
            **self.alt_props
        ).interactive()

        return chart
    
    def tsne(self, n_iter=1000, random_state=None):
        embeddings = self.embeddings

        with warnings.catch_warnings(category=UserWarning):
            tsne = TSNE(
                n_components=2,
                random_state=random_state,
                n_iter=n_iter,
            )
            tsne_result = tsne.fit_transform(embeddings)

        tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
        tsne_df['number'] = range(1000)

        chart = alt.Chart(tsne_df).mark_circle(size=60).encode(
            x='t-SNE1:Q',
            y='t-SNE2:Q',
            color=alt.Color('number:Q', scale=alt.Scale(scheme=self.color_scheme)),
            tooltip=['number:Q', 't-SNE1:Q', 't-SNE2:Q']
        ).properties(
            title=f'{self}: t-SNE',
            **self.alt_props
        ).interactive()

        return chart
    
    def umap(self, n_epochs=2000, random_state=None):
        # setting random_state will disable parallelization

        embeddings = self.embeddings

        with warnings.catch_warnings(category=UserWarning):
            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_epochs=n_epochs,
            )
            umap_result = reducer.fit_transform(embeddings)

        umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
        umap_df['number'] = range(1000)

        chart = alt.Chart(umap_df).mark_circle(size=60).encode(
            x='UMAP1:Q',
            y='UMAP2:Q',
            color=alt.Color('number:Q', scale=alt.Scale(scheme=self.color_scheme)),
            tooltip=['number:Q', 'UMAP1:Q', 'UMAP2:Q']
        ).properties(
            title=f'{self}: UMAP',
            **self.alt_props
        ).interactive()

        return chart

def plot_embeddings(model_id):
    loader = EmbeddingsLoader(model_id)
    numeric_visualizer = EmbeddingsViz(loader.numbers(), model_id, 'Number embeddings between 0-999')
    random_visualizer = EmbeddingsViz(loader.random(), model_id, 'Random embeddings', color_scheme='plasma')
    del loader

    return alt.vconcat(
        alt.hconcat(numeric_visualizer.pca(), random_visualizer.pca()),
        # alt.hconcat(numeric_visualizer.tsne(), random_visualizer.tsne()),
        alt.hconcat(numeric_visualizer.umap(), random_visualizer.umap())
    )