from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
from embeddings_analysis import EmbeddingsData, EmbeddingsDimReduction

# Disable max rows limit for Altair
alt.data_transformers.disable_max_rows()

# Default properties for consistent chart styling
default_props = {"width": 700, "height": 500}


@dataclass
class EmbeddingsPCA(EmbeddingsDimReduction):
    reduction: PCA

    @property
    def pca(self) -> PCA:
        return self.reduction

    @classmethod
    def from_embeddings(
        cls, embeddings_data: EmbeddingsData, pca: PCA = None
    ) -> EmbeddingsPCA:
        """Create an instance of EmbeddingsPCA from embeddings data and PCA model."""
        pca = pca or PCA(n_components=100)
        transformed = pca.transform(embeddings_data.data)
        return cls(embeddings_data, pca, transformed)

    def plot_explained_variance(self) -> alt.Chart:
        """Create a chart showing the explained variance per component."""
        ev_df = pd.DataFrame(
            {
                "Component": np.arange(1, len(self.pca.explained_variance_) + 1),
                "ExplainedVariance": self.pca.explained_variance_,
            }
        )

        return (
            alt.Chart(ev_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Component:Q", title="Component Index"),
                y=alt.Y("ExplainedVariance:Q", title="Explained Variance"),
                tooltip=["Component", "ExplainedVariance"],
            )
            .properties(
                title=f"{self.embeddings_data}: Explained Variance Distribution",
                **default_props,
            )
            .interactive()
        )

    def plot_cumulative_variance(self) -> alt.Chart:
        """Create a chart showing the cumulative explained variance."""
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        variance_df = pd.DataFrame(
            {
                "Components": np.arange(1, len(cumulative_variance) + 1),
                "CumulativeVariance": cumulative_variance,
            }
        )

        threshold_df = pd.DataFrame(
            {"Threshold": [0.9], "Label": ["90% Explained Variance"]}
        )

        chart = (
            alt.Chart(variance_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Components:Q", title="Number of Components"),
                y=alt.Y(
                    "CumulativeVariance:Q",
                    title="Cumulative Explained Variance",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=["Components", "CumulativeVariance"],
            )
            .properties(
                title=f"{self.embeddings_data}: Cumulative Explained Variance",
                **default_props,
            )
            .interactive()
        )

        threshold_rule = (
            alt.Chart(threshold_df)
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y="Threshold:Q", tooltip=["Label"])
        )

        return chart + threshold_rule

    def plot_2d_projection(
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
                title=f"{self.embeddings_data}: PCA Component Projection",
                **default_props,
            )
            .interactive()
        )

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
                title=f"{self.embeddings_data}: Distances Between Consecutive Number Embeddings",
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
            .properties(
                title=f"{self.embeddings_data}: Component Patterns", **default_props
            )
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

    def plot_correlation_heatmap(self, n_vectors: int = 20) -> alt.Chart:
        """Create a heatmap showing correlations between the top components."""
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

        return (
            alt.Chart(pd.DataFrame(corr_data))
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
            .properties(
                title=f"{self.embeddings_data}: Correlations Between Top Principal Components",
                width=600,
                height=600,
            )
        )

    def plot_by_digit(
        self,
        digit_position: int,
        x_component: int = 0,
        y_component: int = 1,
    ) -> alt.Chart:
        """Create a scatter plot colored by digit position (0=ones, 1=tens, 2=hundreds)."""
        if not 0 <= digit_position <= 2:
            raise ValueError("digit_position must be 0 (ones), 1 (tens), or 2 (hundreds)")

        digit_df = pd.DataFrame({
            "Number": range(min(1000, self.transformed.shape[0])),
            f"Component{x_component+1}": self.transformed[:min(1000, self.transformed.shape[0]), x_component],
            f"Component{y_component+1}": self.transformed[:min(1000, self.transformed.shape[0]), y_component],
            "Digit": [get_digit(n, digit_position) for n in range(min(1000, self.transformed.shape[0]))]
        })
        
        position_labels = {0: "Ones", 1: "Tens", 2: "Hundreds"}

        return (
            alt.Chart(digit_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"Component{x_component+1}:Q", title=f"Component {x_component+1}"),
                y=alt.Y(f"Component{y_component+1}:Q", title=f"Component {y_component+1}"),
                color=alt.Color("Digit:N", title=f"{position_labels[digit_position]} Digit"),
                tooltip=["Number", "Digit", f"Component{x_component+1}", f"Component{y_component+1}"]
            )
            .properties(
                title=f"{self.embeddings_data}: Embeddings Colored by {position_labels[digit_position]} Digit",
                **default_props,
            )
            .interactive()
        )

    def plot_by_digit_length(
        self, x_component: int = 0, y_component: int = 1
    ) -> alt.Chart:
        """Create a scatter plot comparing single, double, and triple digit numbers."""
        digit_length_df = pd.DataFrame(
            {
                "Number": range(min(1000, self.transformed.shape[0])),
                "Component1": self.transformed[
                    : min(1000, self.transformed.shape[0]), x_component
                ],
                "Component2": self.transformed[
                    : min(1000, self.transformed.shape[0]), y_component
                ],
                "DigitLength": [
                    "Single" if n < 10 else "Double" if n < 100 else "Triple"
                    for n in range(min(1000, self.transformed.shape[0]))
                ],
            }
        )

        return (
            alt.Chart(digit_length_df)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("Component1:Q", title=f"Component {x_component + 1}"),
                y=alt.Y("Component2:Q", title=f"Component {y_component + 1}"),
                color=alt.Color("DigitLength:N", title="Digit Length"),
                size=alt.Size(
                    "DigitLength:N", scale=alt.Scale(range=[200, 100, 30]), legend=None
                ),
                tooltip=["Number", "DigitLength", "Component1", "Component2"],
            )
            .properties(
                title=f"{self.embeddings_data}: Comparison of Single, Double, and Triple Digit Numbers",
                **default_props,
            )
            .interactive()
        )

    def plot_special_numbers(
        self,
        special_numbers: list[int] = None,
        x_component: int = 0,
        y_component: int = 1,
    ) -> alt.Chart:
        """Create a scatter plot highlighting special numbers."""
        if special_numbers is None:
            special_numbers = [0, 1, 10, 100, 42, 69, 314, 404, 500, 666, 911, 999]

        special_numbers = [n for n in special_numbers if n < self.transformed.shape[0]]

        df = pd.DataFrame(
            {
                "Number": range(min(1000, self.transformed.shape[0])),
                "Component1": self.transformed[
                    : min(1000, self.transformed.shape[0]), x_component
                ],
                "Component2": self.transformed[
                    : min(1000, self.transformed.shape[0]), y_component
                ],
                "DigitLength": [
                    "Single" if n < 10 else "Double" if n < 100 else "Triple"
                    for n in range(min(1000, self.transformed.shape[0]))
                ],
                "IsSpecial": [
                    n in special_numbers
                    for n in range(min(1000, self.transformed.shape[0]))
                ],
            }
        )

        df["Label"] = df["Number"].apply(
            lambda x: str(x) if x in special_numbers else ""
        )

        base_scatter = (
            alt.Chart(df)
            .mark_circle(opacity=0.5)
            .encode(
                x=alt.X("Component1:Q", title=f"Component {x_component + 1}"),
                y=alt.Y("Component2:Q", title=f"Component {y_component + 1}"),
                color=alt.condition(
                    alt.datum.IsSpecial,
                    alt.value("red"),
                    alt.Color("DigitLength:N", scale=alt.Scale(scheme="category10")),
                ),
                size=alt.condition(alt.datum.IsSpecial, alt.value(150), alt.value(30)),
                tooltip=["Number", "IsSpecial", "Component1", "Component2"],
            )
        )

        text_labels = (
            alt.Chart(df[df["IsSpecial"]])
            .mark_text(align="left", baseline="middle", dx=15)
            .encode(x="Component1:Q", y="Component2:Q", text="Label:N")
        )

        return (
            (base_scatter + text_labels)
            .properties(
                title=f"{self.embeddings_data}: Special Numbers in Embedding Space",
                **default_props,
            )
            .interactive()
        )


def pca_decomposition(
    embeddings_data: EmbeddingsData, n_components: int = 100
) -> EmbeddingsPCA:
    """Perform PCA decomposition on embedding data."""
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(embeddings_data.data)
    return EmbeddingsPCA(embeddings_data, pca, transformed)


def get_digit(num: int, position: int) -> int:
    """Get the digit at a specific position (0=ones, 1=tens, 2=hundreds)."""
    return (num // 10**position) % 10


def prepare_digit_data(transformed: np.ndarray, n_components: int = 3) -> pd.DataFrame:
    """Prepare a DataFrame with digit information for visualization."""
    n_components = min(n_components, transformed.shape[1])
    digit_data = []

    for num in range(min(1000, transformed.shape[0])):
        # Get digits at different positions
        ones = get_digit(num, 0)
        tens = get_digit(num, 1)
        hundreds = get_digit(num, 2)

        # Get the top components for this number
        components = transformed[num, :n_components]

        data_point = {
            "Number": num,
            "OnesDigit": ones,
            "TensDigit": tens,
            "HundredsDigit": hundreds,
        }

        # Add components
        for i in range(n_components):
            data_point[f"Component{i + 1}"] = components[i]

        digit_data.append(data_point)

    return pd.DataFrame(digit_data)


def analyze_embeddings(
    embeddings_data: EmbeddingsData,
    n_components: int = 100,
    special_numbers: list[int] = None,
    facet_components: bool = False,
) -> dict[str, alt.Chart]:
    """Perform comprehensive PCA analysis on embeddings and return all visualizations."""
    pca_result = pca_decomposition(embeddings_data, n_components)

    return {
        "explained_variance": pca_result.plot_explained_variance(),
        "cumulative_variance": pca_result.plot_cumulative_variance(),
        "projection": pca_result.plot_2d_projection(),
        "consecutive_distances": pca_result.plot_consecutive_distances(),
        "component_patterns": pca_result.plot_component_patterns(
            facet=facet_components
        ),
        "correlation_heatmap": pca_result.plot_correlation_heatmap(),
        "ones_digit": pca_result.plot_by_digit(0),
        "tens_digit": pca_result.plot_by_digit(1),
        "hundreds_digit": pca_result.plot_by_digit(2),
        "digit_length": pca_result.plot_by_digit_length(),
        "special_numbers": pca_result.plot_special_numbers(special_numbers),
    }
