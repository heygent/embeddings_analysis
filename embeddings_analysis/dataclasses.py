from collections.abc import Iterable
from dataclasses import dataclass, astuple
from functools import cached_property, wraps

import altair as alt
import numpy as np
import pandas as pd

import marimo as mo


@dataclass(eq=True, frozen=True)
class EmbeddingsMeta:
    model_id: str

    @property
    def id(self) -> str:
        return "_".join([
            self.model_id.replace("/", "__"),
            self.__class__.__name__.rstrip("Meta")
        ])

    @property
    def label(self) -> str:
        return f"({self.model_id})"
    
    def dim_reduction(self, estimator):
        return EmbeddingsDimReduction(self, estimator, estimator.fit_transform(self.data))
    

@dataclass(frozen=True, eq=True)
class RangeEmbeddingsMeta(EmbeddingsMeta):
    stop: int = None

    @property
    def label(self) -> str:
        return f"{super().label} Number embeddings 0-{self.stop}"


@dataclass(frozen=True, eq=True)
class RandomEmbeddingsMeta(EmbeddingsMeta):
    sample_size: int
    seed: int

    @property
    def id(self):
        return "_".join([
            super().id,
            str(self.sample_size),
            str(self.seed)
        ])

    @property
    def label(self) -> str:
        return f"{super().label} Random embeddings n={self.sample_size}"


@dataclass(frozen=True, eq=True, init=False)
class ReducedEmbeddingsMeta(EmbeddingsMeta):
    original: EmbeddingsMeta
    estimator: object

    def __init__(self, original: EmbeddingsMeta, estimator: object, data: pd.DataFrame):
        super().__init__(model_id=original.model_id, data=data)
        self.original = original
        self.estimator = estimator


@dataclass(frozen=True, eq=True)
class EmbeddingsData[T: EmbeddingsMeta]:
    meta: T
    embeddings: pd.DataFrame

    @property
    def model_id(self) -> str:
        return self.meta.model_id
    
    @property
    def label(self) -> str:
        return self.meta.label

    def __str__(self):
        return f"({self.model_id}) {self.label}"
    
    @cached_property
    def numpy(self):
        return self.embeddings.drop(['Token', 'Token ID'], axis=1).to_numpy()
    
    def dim_reduction(self, estimator):
        return EmbeddingsDimReduction(self, estimator, estimator.fit_transform(self.numpy))


if mo.running_in_notebook():
    default_props = {}
else:
    default_props = {"width": 450, "height": 400}


def plot_fn(func):
    if mo.running_in_notebook():
        @wraps(func)
        def wrapper(*args, **kwargs):
            return mo.ui.altair_chart(func(*args, **kwargs))
        return wrapper
    
    return func

def patch_plot_methods(obj):
    def patch_plot(orig_method):
        @wraps(orig_method)
        def new_plot(self, *args, **kwargs):
            chart = orig_method(self, *args, **kwargs)
            return mo.ui.altair_chart(chart)
        return new_plot

    for attr in dir(obj):
        if attr.startswith("plot_"):
            method = getattr(obj, attr)
            setattr(obj, attr, patch_plot(method))


@dataclass
class EmbeddingsDimReduction:
    embeddings_data: EmbeddingsData
    estimator: object
    transformed: np.ndarray

    def __str__(self):
        return f"({self.embeddings_data.model_id}) {self.embeddings_data.label} {self.estimator.__class__.__name__}"

    def __post_init__(self):
        if mo.running_in_notebook():
            patch_plot_methods(self)

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.transformed,
            columns=(f'Component{i + 1}' for i in range(self.transformed.shape[1])),
        ).reset_index(names='Number')
    

    def alt_title(self, **kwargs) -> str:
        return alt.TitleParams(
            text=self.embeddings_data.model_id,
            fontSize=24,
            subtitle=f"{self.embeddings_data.label}: {self.estimator.__class__.__name__}",
            subtitleFontSize=16,
            anchor="middle",
            **kwargs,
        )

    def plot(
        self, x_component: int = 0, y_component: int = 1, colorscheme: str = "viridis"
    ) -> alt.Chart:
        """Create a 2D scatter plot of two principal components."""
        return (
            alt.Chart(self.df)
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
                    scale=alt.Scale(scheme=colorscheme),  # type: ignore
                    title="Number Value",
                ),
                tooltip=[
                    "Number",
                    f"Component{x_component + 1}",
                    f"Component{y_component + 1}",
                ],
            )
            .properties(
                title=self.alt_title(),
                **default_props,
            )
            .interactive()
        )

    def plot_digit_overview(self, x_component=0, y_component=1) -> alt.Chart:
        """Create a 2x2 grid of scatter plots for each digit position and digit length."""
        return alt.vconcat(
            alt.hconcat(
                self.plot_by_digit(0, x_component, y_component),
                self.plot_by_digit(1, x_component, y_component),
            ).resolve_scale(color="independent"),
            alt.hconcat(
                self.plot_by_digit(2, x_component, y_component),
                self.plot_by_digit_length(x_component, y_component),
            ).resolve_scale(color="independent"),
        ).properties(title=self.alt_title())

    def plot_by_digit(
        self,
        digit_position: int,
        x_component: int = 0,
        y_component: int = 1,
    ) -> alt.Chart:
        """Create a scatter plot colored by digit position (0=ones, 1=tens, 2=hundreds)."""
        
        match digit_position:
            case 0:
                position_label = "Ones"
            case 1:
                position_label = "Tens"
            case 2:
                position_label = "Hundreds"
            case _:
                raise ValueError("digit_position must be 0 (ones), 1 (tens), or 2 (hundreds)")

        return (
            alt.Chart(self.df)
            .mark_circle(size=60, opacity=0.7)
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
                    "Digit:N", title=f"{position_label} Digit"
                ),
                tooltip=[
                    "Number",
                    "Digit:N",
                    f"Component{x_component + 1}",
                    f"Component{y_component + 1}",
                ],
            )
            .transform_calculate(
                Digit=f"floor(datum.Number / pow(10, {digit_position})) % 10",
            )
            .properties(
                title=f"Embeddings by {position_label} Digit",
                **default_props,
            )
            .interactive()
        )

    def plot_by_digit_length(
        self, x_component: int = 0, y_component: int = 1
    ) -> alt.Chart:
        """Create a scatter plot comparing single, double, and triple digit numbers."""
        return (
            alt.Chart(self.df)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("Component1:Q", title=f"Component {x_component + 1}"),
                y=alt.Y("Component2:Q", title=f"Component {y_component + 1}"),
                color=alt.Color("DigitLength:N", title="Digit Length"),
                size=alt.Size(
                    "DigitLength:N", scale=alt.Scale(range=[200, 100, 30]), legend=None
                ),
                tooltip=["Number", "DigitLength:N", "Component1", "Component2"],
            )
            .transform_calculate(
                DigitLength="length(toString(datum.Number))",
            )
            .properties(
                title="Comparison by Digit Length",
                **default_props,
            )
            .interactive()
        )
    
    @cached_property
    def variance_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Component": np.arange(1, len(self.estimator.explained_variance_) + 1),
                "ExplainedVariance": self.estimator.explained_variance_,
                "ExplainedVarianceRatio": self.estimator.explained_variance_ratio_,
            }
        )

    def plot_explained_variance(self) -> alt.Chart:
        """Create a chart showing the explained variance per component."""
        return (
            alt.Chart(self.variance_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Component:Q", title="Component Index"),
                y=alt.Y("ExplainedVariance:Q", title="Explained Variance"),
                tooltip=["Component", "ExplainedVariance"],
            )
            .properties(
                title="Explained Variance Distribution",
                **default_props,
            )
            .interactive()
        )

    def plot_cumulative_variance(self) -> alt.Chart:
        """Create a chart showing the cumulative explained variance."""
        chart = (
            alt.Chart(self.variance_df)
            .transform_window(
                CumulativeVariance="sum(ExplainedVarianceRatio)",
                Components="count()",
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("Components:Q", title="Number of Components"),
                y=alt.Y(
                    "CumulativeVariance:Q",
                    title="Cumulative Explained Variance",
                ).scale(domain=[0, 1]).axis(format=".0%"),
                tooltip=["Components:Q", "CumulativeVariance:Q"],
            )
            .properties(
                title="Cumulative Explained Variance",
                **default_props,
            )
            .interactive(bind_y=False)
        )

        threshold_rule = (
            alt.Chart()
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y=alt.Y(datum=0.9))
        )

        return chart + threshold_rule

    def plot_variance_overview(self) -> alt.Chart:
        return alt.hconcat(
            self.plot_explained_variance(), self.plot_cumulative_variance()
        ).properties(title=self.alt_title())

    def plot_consecutive_distances(self) -> alt.Chart:
        """Create a chart showing distances between consecutive number embeddings."""
        consecutive_distances = np.linalg.norm(
            self.embeddings_data.data[1:] - self.embeddings_data.data[:-1], axis=1
        )
        distances_df = pd.DataFrame(
            {
                "Number": np.arange(1, len(consecutive_distances) + 1),
                "Distance": consecutive_distances,
            }
        )

        return (
            alt.Chart(distances_df)
            .mark_line()
            .encode(
                x=alt.X("Number:Q", title="Number n (distance is between n-1 and n)"),
                y=alt.Y("Distance:Q", title="Euclidean Distance"),
                tooltip=["Number", "Distance"],
            )
            .properties(
                title="Distances Between Consecutive Numbers",
                **default_props,
            )
            .interactive()
        )

    def plot_component_patterns(
        self, n_components: int = 5, n_values: int = 100, facet: bool = True
    ) -> alt.Chart:
        """Create a chart showing patterns in the top components."""
        component_dfs = []
        for i in range(min(n_components, len(self.pca.components_))):
            df = pd.DataFrame(
                {
                    "Index": np.arange(min(n_values, len(self.pca.components_[i]))),
                    "Value": self.pca.components_[
                        i, : min(n_values, len(self.pca.components_[i]))
                    ],
                    "Component": f"Component {i + 1}",
                }
            )
            component_dfs.append(df)

        component_df = pd.concat(component_dfs)
        base_chart = (
            alt.Chart(component_df)
            .mark_line()
            .encode(
                x=alt.X("Index:Q", title="Index"),
                color=alt.Color("Component:N", title="Component"),
                tooltip=["Component", "Index", "Value"],
            )
            .properties(title="Component Patterns", **default_props)
            .interactive()
        )

        return (
            (
                base_chart.encode(y=alt.Y("Value:Q", title="Value"))
                .facet(row="Component:N")
                .resolve_scale(y="independent")
            )
            if facet
            else base_chart.encode(y=alt.Y("Value:Q", title="Value"))
        )

    def top_correlations_df(
        self, n_vectors: int = 20, min_correlation=0.01
    ) -> pd.DataFrame:
        n_vectors = min(n_vectors, self.transformed.shape[1])
        correlations = np.corrcoef(self.transformed[:, :n_vectors].T)

        # Get upper triangle indices (excluding diagonal)
        idx_i, idx_j = np.triu_indices(n_vectors, k=1)
        data = {
            "Component1": idx_i,
            "Component2": idx_j,
            "Correlation": correlations[idx_i, idx_j],
        }
        corr_df = pd.DataFrame(data)

        # Sort by absolute correlation descending
        corr_df = corr_df.reindex(
            corr_df["Correlation"].abs().sort_values(ascending=False).index
        )
        corr_df = corr_df[["Component1", "Component2", "Correlation"]]
        return corr_df[corr_df["Correlation"].abs() >= min_correlation]

    def plot_top_correlated_components(self, n_vectors: int = 10) -> alt.Chart:
        """
        Plot the plot_by_digit_length chart for the top correlated component pairs.
        Arranges the plots in a grid with two plots per row.
        """
        corr_df = self.top_correlations_df(n_vectors)
        charts = []
        for _, row in corr_df.iterrows():
            i, j = int(row["Component1"]), int(row["Component2"])
            chart = self.plot(x_component=i, y_component=j).properties(
                title=f"Components {i + 1} vs {j + 1} (corr={row['Correlation']:.2f})"
            )
            charts.append(chart)
            chart = self.plot_by_digit_length(x_component=i, y_component=j).properties(
                title=f"Components {i + 1} vs {j + 1} (corr={row['Correlation']:.2f}) by digit length"
            )
            charts.append(chart)
        # Arrange charts in a grid with two plots per row
        rows = []
        for k in range(0, len(charts), 2):
            row = alt.hconcat(*charts[k : k + 2])
            rows.append(row)
        return alt.vconcat(*rows).properties(
            title=self.alt_title(),
        )

    def plot_correlation_heatmap(self, n_vectors: int = 20) -> alt.Chart:
        """Create a heatmap showing correlations between the top components, with value labels."""
        n_vectors = min(n_vectors, self.transformed.shape[1])
        correlations = np.corrcoef(self.transformed[:, :n_vectors].T)

        corr_data = [
            {
                "Component1": i + 1,
                "Component2": j + 1,
                "Correlation": correlations[i, j],
            }
            for i in range(n_vectors)
            for j in range(n_vectors)
        ]
        df = pd.DataFrame(corr_data)

        heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("Component1:O", title="Component"),
                y=alt.Y("Component2:O", title="Component"),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                ),
                tooltip=["Component1", "Component2", "Correlation"],
            )
        )

        text = (
            alt.Chart(df)
            .mark_text(baseline="middle", fontSize=12)
            .encode(
                x=alt.X("Component1:O"),
                y=alt.Y("Component2:O"),
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "abs(datum.Correlation) > 0.5",
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        return (
            (heatmap + text)
            .properties(
                title=self.alt_title(),
                width=600,
                height=600,
            )
        )