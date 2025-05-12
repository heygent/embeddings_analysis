import numpy as np
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
import torch

from embeddings_analysis import EmbeddingsData

# Disable max rows limit for Altair
alt.data_transformers.disable_max_rows()

# Default properties for consistent chart styling
default_props = {
    'width': 700,
    'height': 500
}

def plot_singular_values(embeddings_data: EmbeddingsData) -> alt.Chart:
    """
    Create a chart showing the singular value distribution from PCA.
    """
    sv_df = pd.DataFrame({
        'Component': np.arange(1, len(embeddings_data.pca.singular_values_) + 1),
        'SingularValue': embeddings_data.pca.singular_values_
    })

    chart = alt.Chart(sv_df).mark_line(point=True).encode(
        x=alt.X('Component:Q', title='Component Index'),
        y=alt.Y('SingularValue:Q', title='Singular Value'),
        tooltip=['Component', 'SingularValue']
    ).properties(
        title=f'{embeddings_data}: Singular Value Distribution',
        **default_props
    ).interactive()
    
    return chart

def plot_cumulative_variance(embeddings_data: EmbeddingsData) -> alt.Chart:
    """
    Create a chart showing the cumulative explained variance from PCA.
    """
    explained_variance_ratio = embeddings_data.pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    variance_df = pd.DataFrame({
        'Components': np.arange(1, len(cumulative_variance) + 1),
        'CumulativeVariance': cumulative_variance
    })

    threshold_df = pd.DataFrame({
        'Threshold': [0.9],
        'Label': ['90% Explained Variance']
    })

    chart = alt.Chart(variance_df).mark_line(point=True).encode(
        x=alt.X('Components:Q', title='Number of Components'),
        y=alt.Y('CumulativeVariance:Q', title='Cumulative Explained Variance', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Components', 'CumulativeVariance']
    ).properties(
        title=f'{embeddings_data}: Cumulative Explained Variance',
        **default_props
    ).interactive()

    threshold_rule = alt.Chart(threshold_df).mark_rule(color='red', strokeDash=[4, 4]).encode(
        y='Threshold:Q',
        tooltip=['Label']
    )

    return chart + threshold_rule

def plot_pca_2d_projection(embeddings_data: EmbeddingsData, 
                           component1: int = 0, component2: int = 1, 
                           colorscheme: str = 'viridis') -> alt.Chart:
    """
    Create a 2D scatter plot of two principal components from PCA.
    """

    transformed = embeddings_data.pca_result
    projection_df = pd.DataFrame({
        f'Component{component1+1}': transformed[:, component1],
        f'Component{component2+1}': transformed[:, component2],
        'Number': np.arange(transformed.shape[0])
    })

    chart = alt.Chart(projection_df).mark_circle(size=60).encode(
        x=alt.X(f'Component{component1+1}:Q', title=f'Component {component1+1}'),
        y=alt.Y(f'Component{component2+1}:Q', title=f'Component {component2+1}'),
        color=alt.Color('Number:Q', scale=alt.Scale(scheme=colorscheme), title='Number Value'),
        tooltip=['Number', f'Component{component1+1}', f'Component{component2+1}']
    ).properties(
        title=f'{embeddings_data}: PCA Component Projection',
        **default_props
    ).interactive()
    
    return chart

def plot_consecutive_distances(embeddings_data: EmbeddingsData) -> alt.Chart:
    """
    Create a chart showing distances between consecutive number embeddings.
    """
    data = embeddings_data.data
    
    consecutive_distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
    distances_df = pd.DataFrame({
        'Number': np.arange(1, len(consecutive_distances) + 1),
        'Distance': consecutive_distances
    })

    chart = alt.Chart(distances_df).mark_line().encode(
        x=alt.X('Number:Q', title='Number n (distance is between n-1 and n)'),
        y=alt.Y('Distance:Q', title='Euclidean Distance'),
        tooltip=['Number', 'Distance']
    ).properties(
        title=f'{embeddings_data}: Distances Between Consecutive Number Embeddings',
        **default_props
    ).interactive()
    
    return chart

def plot_principal_component_patterns(embeddings_data: EmbeddingsData, 
                                n_components: int = 5, n_values: int = 100) -> alt.Chart:
    """
    Create a chart showing patterns in the top PCA components.
    """
    component_dfs = []
    for i in range(min(n_components, len(embeddings_data.pca.components_))):
        df = pd.DataFrame({
            'Index': np.arange(min(n_values, len(embeddings_data.pca.components_[i]))),
            'Value': embeddings_data.pca.components_[i, :min(n_values, len(embeddings_data.pca.components_[i]))],
            'Component': f'Component {i+1}'
        })
        component_dfs.append(df)

    component_df = pd.concat(component_dfs)

    chart = alt.Chart(component_df).mark_line().encode(
        x=alt.X('Index:Q', title='Index'),
        y=alt.Y('Value:Q', title='Value'),
        color='Component:N',
        tooltip=['Component', 'Index', 'Value']
    ).properties(
        title=f'{embeddings_data}: PCA Component Patterns',
        **default_props
    ).facet(
        row='Component:N'
    ).resolve_scale(
        y='independent'
    ).interactive()
    
    return chart

def plot_correlation_heatmap(embeddings_data: EmbeddingsData, n_vectors: int = 20) -> alt.Chart:
    """
    Create a heatmap showing correlations between the top components.
    """
    transformed = embeddings_data.pca_result
    n_vectors = min(n_vectors, transformed.shape[1])
    correlations = np.corrcoef(transformed[:, :n_vectors].T)

    corr_data = []
    for i in range(n_vectors):
        for j in range(n_vectors):
            corr_data.append({
                'Component1': i+1,
                'Component2': j+1,
                'Correlation': correlations[i, j]
            })

    corr_df = pd.DataFrame(corr_data)

    chart = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('Component1:O', title='Component'),
        y=alt.Y('Component2:O', title='Component'),
        color=alt.Color('Correlation:Q', 
                      scale=alt.Scale(scheme='blueorange', domain=[-1, 1])),
        tooltip=['Component1', 'Component2', 'Correlation']
    ).properties(
        title=f'{embeddings_data}: Correlations Between Top Singular Vectors',
        width=600,
        height=600
    )
    
    return chart

def get_digit(num: int, position: int) -> int:
    """
    Get the digit at a specific position.
    
    Args:
        num: The number to extract the digit from
        position: Position (0=ones, 1=tens, 2=hundreds)
        
    Returns:
        The digit at the specified position
    """
    return (num // 10**position) % 10

def prepare_digit_data(embeddings_data: EmbeddingsData, n_components: int = 3) -> pd.DataFrame:
    """
    Prepare a DataFrame with digit information for visualization.
    """
    transformed = embeddings_data.pca_result
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
            'Number': num,
            'OnesDigit': ones,
            'TensDigit': tens,
            'HundredsDigit': hundreds
        }
        
        # Add components
        for i in range(n_components):
            data_point[f'Component{i+1}'] = components[i]
            
        digit_data.append(data_point)
    
    return pd.DataFrame(digit_data)

def plot_by_digit(embeddings_data: EmbeddingsData, digit_df: pd.DataFrame, 
                 digit_type: str, component1: int = 1, component2: int = 2) -> alt.Chart:
    """
    Create a scatter plot colored by a specific digit position.
    """
    # Validate digit type
    valid_types = ['OnesDigit', 'TensDigit', 'HundredsDigit']
    if digit_type not in valid_types:
        raise ValueError(f"digit_type must be one of {valid_types}")
    
    # Create readable label for title
    label_map = {
        'OnesDigit': 'Ones Digit',
        'TensDigit': 'Tens Digit', 
        'HundredsDigit': 'Hundreds Digit'
    }
    
    chart = alt.Chart(digit_df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X(f'Component{component1}:Q', title=f'Component {component1}'),
        y=alt.Y(f'Component{component2}:Q', title=f'Component {component2}'),
        color=alt.Color(f'{digit_type}:N', title=label_map[digit_type]),
        tooltip=['Number', 'OnesDigit', 'TensDigit', 'HundredsDigit', f'Component{component1}', f'Component{component2}']
    ).properties(
        title=f'{embeddings_data}: Embeddings Colored by {label_map[digit_type]}',
        **default_props
    ).interactive()
    
    return chart

def plot_by_digit_length(embeddings_data: EmbeddingsData, 
                        component1: int = 0, component2: int = 1) -> alt.Chart:
    """
    Create a scatter plot comparing single, double, and triple digit numbers.
    """
    transformed = embeddings_data.pca_result
    digit_length_df = pd.DataFrame({
        'Number': range(min(1000, transformed.shape[0])),
        'Component1': transformed[:min(1000, transformed.shape[0]), component1],
        'Component2': transformed[:min(1000, transformed.shape[0]), component2],
        'DigitLength': ['Single' if n < 10 else 'Double' if n < 100 else 'Triple' for n in range(min(1000, transformed.shape[0]))]
    })

    chart = alt.Chart(digit_length_df).mark_circle(opacity=0.7).encode(
        x=alt.X('Component1:Q', title=f'Component {component1+1}'),
        y=alt.Y('Component2:Q', title=f'Component {component2+1}'),
        color=alt.Color('DigitLength:N', title='Digit Length'),
        size=alt.Size('DigitLength:N', 
                     scale=alt.Scale(range=[200, 100, 30]),
                     legend=None),
        tooltip=['Number', 'DigitLength', 'Component1', 'Component2']
    ).properties(
        title=f'{embeddings_data}: Comparison of Single, Double, and Triple Digit Numbers',
        **default_props
    ).interactive()
    
    return chart

def plot_special_numbers(embeddings_data: EmbeddingsData, 
                       special_numbers: list[int] = None, component1: int = 0, component2: int = 1) -> alt.Chart:
    """
    Create a scatter plot highlighting special numbers.
    """
    transformed = embeddings_data.pca_result
    if special_numbers is None:
        special_numbers = [0, 1, 10, 100, 42, 69, 314, 404, 500, 666, 911, 999]
    
    # Filter out numbers outside our range
    special_numbers = [n for n in special_numbers if n < transformed.shape[0]]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Number': range(min(1000, transformed.shape[0])),
        'Component1': transformed[:min(1000, transformed.shape[0]), component1],
        'Component2': transformed[:min(1000, transformed.shape[0]), component2],
        'DigitLength': ['Single' if n < 10 else 'Double' if n < 100 else 'Triple' for n in range(min(1000, transformed.shape[0]))],
        'IsSpecial': [n in special_numbers for n in range(min(1000, transformed.shape[0]))]
    })
    
    # Add labels for special numbers
    df['Label'] = df['Number'].apply(lambda x: str(x) if x in special_numbers else '')
    
    # Base scatter plot
    base_scatter = alt.Chart(df).mark_circle(opacity=0.5).encode(
        x=alt.X('Component1:Q', title=f'Component {component1+1}'),
        y=alt.Y('Component2:Q', title=f'Component {component2+1}'),
        color=alt.condition(
            alt.datum.IsSpecial,
            alt.value('red'),
            alt.Color('DigitLength:N', scale=alt.Scale(scheme='category10'))
        ),
        size=alt.condition(
            alt.datum.IsSpecial,
            alt.value(150),
            alt.value(30)
        ),
        tooltip=['Number', 'IsSpecial', 'Component1', 'Component2']
    )
    
    # Text labels for special numbers
    text_labels = alt.Chart(df[df['IsSpecial']]).mark_text(
        align='left',
        baseline='middle',
        dx=15
    ).encode(
        x='Component1:Q',
        y='Component2:Q',
        text='Label:N'
    )
    
    # Combined chart
    chart = (base_scatter + text_labels).properties(
        title=f'{embeddings_data}: Special Numbers in Embedding Space',
        **default_props
    ).interactive()
    
    return chart

# Complete analysis function that returns all charts
def analyze_embeddings(embeddings_data: EmbeddingsData, n_components: int = 100, 
                      special_numbers: list[int] = None) -> dict[str, alt.Chart]:
    """
    Perform comprehensive PCA analysis on embeddings and return all visualizations.
    
    Args:
        embeddings_data: The embedding data to analyze
        n_components: Number of PCA components to extract
        special_numbers: List of numbers to highlight in special numbers plot
        
    Returns:
        Dictionary of named charts
    """
    # Prepare digit data
    digit_df = prepare_digit_data(embeddings_data)
    
    # Create all charts
    charts = {
        'singular_values': plot_singular_values(embeddings_data),
        'cumulative_variance': plot_cumulative_variance(embeddings_data),
        'projection': plot_pca_2d_projection(embeddings_data),
        'consecutive_distances': plot_consecutive_distances(embeddings_data),
        'component_patterns': plot_principal_component_patterns(embeddings_data),
        'correlation_heatmap': plot_correlation_heatmap(embeddings_data),
        'ones_digit': plot_by_digit(embeddings_data, digit_df, 'OnesDigit'),
        'tens_digit': plot_by_digit(embeddings_data, digit_df, 'TensDigit'),
        'hundreds_digit': plot_by_digit(embeddings_data, digit_df, 'HundredsDigit'),
        'digit_length': plot_by_digit_length(embeddings_data),
        'special_numbers': plot_special_numbers(embeddings_data, special_numbers)
    }
    
    return charts