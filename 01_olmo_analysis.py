

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    import altair as alt
    from embeddings_analysis.loader import get_loader

    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from umap import UMAP

    import warnings

    alt.data_transformers.disable_max_rows()
    alt.renderers.set_embed_options(theme="dark")
    return PCA, TSNE, TruncatedSVD, UMAP, alt, get_loader, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Numeric embedding analysis - OLMo-2-1124-7B

        The model has been chosen as one of the targets of this analysis because of its inclination towards research and hackability. Like the other models considered, OLMo uses a BPE tokenizer in which the 0-999 range seems to be hardcoded to be encoded with a single token for each number.
        Only numbers in this range are considered, even though there might be bigger integers that get encoded with a single token by the BPE tokenizer.

        It is also notable that the OLMo model shares a lot of similarities with the LLaMa

        We check the numbers that get encoded in a single embedding vector by running the tokenizer on all the numbers in the range until we find the first one that gets encoded with more than one token.
        """
    )
    return


@app.cell
def _(get_loader):
    model_id = "allenai/OLMo-2-1124-7B"
    loader = get_loader(model_id)
    return (loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One of the goal of this analysis is to find structures in the embeddings.

        Hypotheses:
        - The representation of different models converges to similar structures
        - Numerical embeddings provide a representation that favors numerical calculation tasks
        - There are structures that the embeddings converge towards in the pursuit of certain tasks
            - Some structure provide [affordances](https://en.wikipedia.org/wiki/Affordance) that allow for better resolution of certain tasks.

        It's also notable that the choice of having specific tokens for the numbers in the range 0-999 bias the model toward a direct representation of positive integers, possibly negating symmetries with negative numbers.

        TODO Confront this with [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) and other models
        """
    )
    return


@app.cell
def _(loader):
    # Loading the number embeddings and 1000 random embeddings for comparison

    number_embeddings = loader.range()
    number_embeddings.embeddings
    return (number_embeddings,)


@app.cell
def _(loader):
    random_embeddings = loader.random()
    random_embeddings.embeddings
    return (random_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Dimensionality reduction techniques are employed to visualize the structures that might emerge from the embeddings. They also are compared to a visualization of random embeddings to show that the structure is specific to the number embeddings.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Linear Dimensionality Reduction
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Principal Component Analysis

        The numerical embeddings form a clear curve, suggesting they might follow a meaningful geometric pattern. The structure might follow this pattern for different reasons:

        - The embdeddings capture non-linear relationships between the number tokens, which may take place in natural language data.
        - PCA tries to preserve large distances in the data, which can cause a "bending" of inherently sequential data when projected to lower dimensions.
        - Curves in PCA might happen because of Guttman effect, see [Camiz](https://www.researchgate.net/publication/228760485_The_Guttman_effect_Its_interpretation_and_a_new_redressing_method)
            - Maybe not, as similar structures appear using just SVD?

        The color gradient is smooth, showing that the embedding space captures numerical proximity. Looking at the top right part of the curve, there looks to be a smear. It might seem incidental, but I'm gonna argue with further visualizations that it represents a recursive encoding of the numbers with one and two digits in the embedding space. Lower numbers find themselves in the right part of the color gradient, and they also happen to be in the right place.
        """
    )
    return


@app.cell
def _(PCA, alt, number_embeddings, random_embeddings):
    number_pca = number_embeddings.dim_reduction(PCA(n_components=1000))
    random_pca = random_embeddings.dim_reduction(PCA(n_components=1000))

    alt.hconcat(number_pca.plot(), random_pca.plot()).resolve_scale(color="independent")
    return (number_pca,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explained variance

        The explained variance distribution pot shows a sharp elbow drop around the 50 dimensions mark. The cumulative explained variance plot shows how 90% of the variance can be explained by approximatively 600 components, suggesting that the intrinsic dimensionality of the numerical embeddings is much lower than the 4096 dimensions provided by the embeddings' size. This gives credit to the hypothesis that the data resides on a lower-dimensional manifold.
        """
    )
    return


@app.cell
def _(number_pca):
    number_pca.plot_variance_overview()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Singular Value Decomposition

        By applying SVD instead of PCA, and avoiding the mean normalization, some even more interesting patterns emerge. By plotting the first two components, the same localized digit clusters appear, but they repeat in a much more consistent manner, suggesting that the model learns the same structure for each cluster of digit counts (one, two and three digit numbers). 

        Avoiding PCA's normalization shows a much clearer structure, which suggests that information may be encoded in the absolute distance from the origin.
        """
    )
    return


@app.cell
def _(TruncatedSVD, alt, number_embeddings, random_embeddings):
    number_svd = number_embeddings.dim_reduction(TruncatedSVD(n_components=100))
    _random_svd = random_embeddings.dim_reduction(TruncatedSVD(n_components=100))
    alt.hconcat(number_svd.plot(), _random_svd.plot()).properties().resolve_scale(color='independent')
    return (number_svd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following plots better show the relationship between digits and their position. Once it gets to the hundreds, the clusters formed are clear, since at that point it would also correspond to a quantitative division in tenths of all the embeddings. The most notable plot is the one that colors by digit length, as it shows very clearly a self-similar repeating structure for each digit count.

        - The structure seems to be fractal, which induces the question on whether this same structure would repeat if higher range numbers would be tokenized singularly (1000-9999 and so on).
            - as much as I want to say fractal, this appears to happen only on this component pair, as following plots show.
        """
    )
    return


@app.cell
def _(number_svd):
    number_svd.plot_digit_overview()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Correlation Heatmap

        The following heatmap shows the correlation between the first 20 SVD components. Some key observations are:

        - Component 1 shows both positive and negative correlations with multiple higher components
            - This suggests component 1 captures a fundamental property that permeates the entire embedding structure
            - Notable correlations between components 1, 2, and 3, particularly the stronger correlation between components 1-2
        - Non-orthogonal structure in early components
            - Numerical structures don't align perfectly with single SVD dimensions but span multiple components
        - Comparison with random embeddings shows similar structure
            - The correlation might relate more with the way embeddings encode language in general rather than specific properties of numeric embeddings
            - However, component 2 seems to VERY CLEARLY encode magnitude? The gradient is too smooth.
        """
    )
    return


@app.cell
def _(TruncatedSVD, alt, number_svd, random_embeddings):
    _random_svd = random_embeddings.dim_reduction(TruncatedSVD(n_components=100))
    alt.hconcat(number_svd.plot_correlation_heatmap(20), _random_svd.plot_correlation_heatmap(20))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Components with maximum correlation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Every component pair shows structure through the gradient and the digit-length clustering, although component pair $(1, 2)$ is the one that shows this difference most cleanly. All plots are against component 1, so they share the x axis while the points are "scrambled" vertically, making the correlation with component 2 all the more interesting given that it is the only one that makes the gradient so smooth.

        Given that the plots are drawn by max correlation, it is likely that they mantain the same likeness, although this triangular structure is mantained even by the $(1,2)$ components of the random embeddings.
        """
    )
    return


@app.cell
def _(number_svd):
    number_svd.plot_top_correlated_components()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Non-Linear Dimensionality Reduction
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## t-SNE
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With t-SNE, there relation captured seems interesting as the gradient seems to rotate in a spiral-like fashion. This would probably the most interesting one to attempt a 3D visualization of.
        """
    )
    return


@app.cell
def _(TSNE, alt, number_embeddings, random_embeddings):
    tsne_kwargs = dict(
        perplexity=75,
        max_iter=3000,
        learning_rate=500,
        early_exaggeration=20,
        random_state=43,
    )

    number_tsne = number_embeddings.dim_reduction(TSNE(**tsne_kwargs))
    random_tsne = random_embeddings.dim_reduction(TSNE(**tsne_kwargs))

    alt.hconcat(number_tsne.plot(), random_tsne.plot()).resolve_scale(color="independent")
    return (number_tsne,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The single and double digit numbers are predominantly towards the center, with some outlier like the number 4 being outside the boundaries of the sphere formed.
        The double digit numbers approximate a curve.
        """
    )
    return


@app.cell
def _(number_tsne):
    number_tsne.plot_digit_overview()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## UMAP
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With UMAP, I make two attempts given the previous observations of the possibility of meaningfulness of the distance between the points and the origin. Cosine similarity is usually used for embeddings, but it doesn't preserve this information, so the visualization is also attempted with Euclidean distance.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Cosine similarity

        The UMAP visualization makes a clear separation of clusters that are quantitatively well separated in the hundreds. The random embeddings show no comparable structure, showing that this is not an artifact. 

        - while clearly reflective of the structure and patterns in the data, this looks strange. It feels like the clustering, while clear and right, might hide some other underlying pattern. However, this is just vibes
        """
    )
    return


@app.cell
def _(UMAP, alt, number_embeddings, random_embeddings, warnings):
    umap_kwargs = dict(
        # Increase from default 15 to preserve more global structure
        n_neighbors=50,        
        # Decrease from default 0.1 for tighter local clusters
        min_dist=0.05,         
        metric="cosine",
        # Increase from default 1.0 to spread out the visualization
        spread=1.5,            
        # Increase to enhance local structure preservation
        local_connectivity=2,  
        random_state=42,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        number_umap_cos = number_embeddings.dim_reduction(UMAP(**umap_kwargs))
        random_umap_cos = random_embeddings.dim_reduction(UMAP(**umap_kwargs))

    alt.hconcat(number_umap_cos.plot(), random_umap_cos.plot()).resolve_scale(color="independent")
    return number_umap_cos, umap_kwargs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are some notable characteristics in the disposition of the numbers: they very systematically go in order from lower (internal) to higher (external).
        It's naturally visible in the tens plot, and zooming in and mouseovering the numbers seem to be approximatively completely sorted.
        - maybe distance from the center not as important as presumed
        """
    )
    return


@app.cell
def _(number_umap_cos):
    number_umap_cos.plot_digit_overview()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Euclidean distance

        It's weird: there's a whirpool where single digit numbers and low hundreds are external, then approximatively at the 200 mark the numbers reappear at the center of the whirpool and spiral out. The random embeddings also seem to be organized as a spiral, so it isn't clear if there isn't some artifact / why is this happening.
        """
    )
    return


@app.cell
def _(UMAP, alt, number_embeddings, random_embeddings, umap_kwargs, warnings):
    umap_kwargs.update(metric='euclidean')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        number_umap_euc = number_embeddings.dim_reduction(UMAP(**umap_kwargs))
        random_umap_euc = random_embeddings.dim_reduction(UMAP(**umap_kwargs))

    alt.hconcat(number_umap_euc.plot(), random_umap_euc.plot()).resolve_scale(color="independent")
    return (number_umap_euc,)


@app.cell
def _(number_umap_euc):
    number_umap_euc.plot_digit_overview()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
