import numpy as np
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
from embeddings_analysis import EmbeddingsData

# Disable max rows limit for Altair
alt.data_transformers.disable_max_rows()

def plot_component_with_numbers(embeddings_data: EmbeddingsData, pca: PCA, 
                               component_idx: int = 2, n_values: int = 100) -> alt.Chart:
    """Create a visualization showing component values with corresponding numbers."""
    # Extract component values
    component_values = pca.components_[component_idx, :n_values]
    
    # Create DataFrame with index values and component values
    df = pd.DataFrame({
        'Index': np.arange(n_values),
        'Value': component_values,
        'Number': np.arange(n_values)  # Actual numbers 0-99
    })
    
    # Create line chart
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Index:Q', title='Number'),
        y=alt.Y('Value:Q', title=f'Component {component_idx+1} Value'),
        tooltip=['Number', 'Value']
    )
    
    # Create text labels for numbers
    text = alt.Chart(df.iloc[::10]).mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        x='Index:Q',
        y='Value:Q',
        text='Number:N'
    )
    
    # Combine line and text
    chart = (line + text).properties(
        title=f'{embeddings_data}: Component {component_idx+1} Pattern with Number Labels',
        width=800,
        height=400
    ).interactive()
    
    return chart

def plot_digit_periodicity(embeddings_data: EmbeddingsData, pca: PCA, 
                          component_idx: int = 2, max_number: int = 100) -> alt.Chart:
    """Create a visualization highlighting digit patterns within a component."""
    # Extract component values
    component_values = pca.components_[component_idx, :max_number]
    
    # Create DataFrame with digits and component values
    df = pd.DataFrame({
        'Number': np.arange(max_number),
        'Value': component_values,
        'OnesDigit': [n % 10 for n in range(max_number)],
        'TensDigit': [(n // 10) % 10 for n in range(max_number)]
    })
    
    # Create base chart
    base = alt.Chart(df).encode(
        tooltip=['Number', 'Value', 'OnesDigit', 'TensDigit']
    ).properties(
        title=f'{embeddings_data}: Component {component_idx+1} Pattern Colored by Digit',
        width=800,
        height=400
    ).interactive()
    
    # Line chart colored by ones digit
    ones_line = base.mark_line().encode(
        x=alt.X('Number:Q', title='Number'),
        y=alt.Y('Value:Q', title=f'Component {component_idx+1} Value'),
        color=alt.Color('OnesDigit:N', title='Ones Digit', 
                      scale=alt.Scale(scheme='category10'))
    )
    
    # Points colored by ones digit
    ones_points = base.mark_circle(size=60).encode(
        x='Number:Q',
        y='Value:Q',
        color='OnesDigit:N'
    )
    
    # Combine
    chart = ones_line + ones_points
    
    return chart

def plot_digit_value_comparison(embeddings_data: EmbeddingsData, pca: PCA, 
                               component_idx: int = 2, max_number: int = 100) -> alt.Chart:
    """Compare component values for numbers with the same ones digit."""
    # Extract component values
    component_values = pca.components_[component_idx, :max_number]
    
    # Create DataFrame with digits and component values
    df = pd.DataFrame({
        'Number': np.arange(max_number),
        'Value': component_values,
        'OnesDigit': [n % 10 for n in range(max_number)]
    })
    
    # Create a chart for each ones digit
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Number:Q', title='Number'),
        y=alt.Y('Value:Q', title=f'Component {component_idx+1} Value'),
        color=alt.Color('OnesDigit:N', title='Ones Digit',
                      scale=alt.Scale(scheme='category10')),
        tooltip=['Number', 'Value', 'OnesDigit']
    ).properties(
        title=f'{embeddings_data}: Component {component_idx+1} Values by Ones Digit',
        width=800,
        height=500
    ).interactive()
    
    return chart

def plot_heatmap_periodicity(embeddings_data: EmbeddingsData, pca: PCA, 
                            component_indices: list[int] = [2, 4], 
                            max_number: int = 100) -> alt.Chart:
    """Create a heatmap showing digit patterns across multiple components."""
    # Extract components
    components = pca.components_[component_indices, :max_number]
    
    # Create a long-format DataFrame
    rows = []
    for i, comp_idx in enumerate(component_indices):
        for n in range(max_number):
            rows.append({
                'Number': n,
                'OnesDigit': n % 10,
                'TensDigit': (n // 10) % 10,
                'Component': f'Component {comp_idx+1}',
                'Value': components[i, n]
            })
    
    df = pd.DataFrame(rows)
    
    # Create heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('OnesDigit:O', title='Ones Digit'),
        y=alt.Y('Component:N', title='Component'),
        color=alt.Color('Value:Q', scale=alt.Scale(scheme='blueorange')),
        tooltip=['Component', 'OnesDigit', 'Value']
    ).properties(
        title=f'{embeddings_data}: Component Values by Ones Digit (Heatmap)',
        width=600,
        height=200
    ).facet(
        row='TensDigit:O'
    )
    
    return chart

def plot_radar_digits(embeddings_data: EmbeddingsData, pca: PCA, 
                     component_indices: list[int] = [0, 1, 2, 3, 4], 
                     digits: list[int] = list(range(10))) -> alt.Chart:
    """Create a radar-like visualization comparing how components encode each digit."""
    # Extract top components
    components = pca.components_[component_indices, :]
    
    # Create a long-format DataFrame for digit patterns
    rows = []
    for i, comp_idx in enumerate(component_indices):
        for digit in digits:
            # Find all numbers with this ones digit
            digit_indices = [n for n in range(min(100, components.shape[1])) if n % 10 == digit]
            
            # Calculate average component value for this digit
            avg_value = np.mean(components[i, digit_indices])
            
            rows.append({
                'Component': f'Component {comp_idx+1}',
                'ComponentIdx': comp_idx + 1,  # For ordering
                'Digit': digit,
                'Value': avg_value
            })
    
    df = pd.DataFrame(rows)
    
    # Create radar-like chart
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Component:N', title='Component', sort=alt.EncodingSortField(field='ComponentIdx')),
        y=alt.Y('Value:Q', title='Average Value'),
        color=alt.Color('Digit:N', title='Digit'),
        detail='Digit:N',
        tooltip=['Component', 'Digit', 'Value']
    ).properties(
        title=f'{embeddings_data}: How Components Encode Each Digit',
        width=600,
        height=400
    ).interactive()
    
    return chart

def plot_component_correlation(embeddings_data: EmbeddingsData, pca: PCA,
                              comp1: int = 2, comp2: int = 4, 
                              max_number: int = 100) -> alt.Chart:
    """Plot correlation between two components, colored by ones digit."""
    # Extract component values
    comp1_values = pca.components_[comp1, :max_number]
    comp2_values = pca.components_[comp2, :max_number]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Number': np.arange(max_number),
        f'Component{comp1+1}': comp1_values,
        f'Component{comp2+1}': comp2_values,
        'OnesDigit': [n % 10 for n in range(max_number)]
    })
    
    # Create scatter plot
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(f'Component{comp1+1}:Q', title=f'Component {comp1+1}'),
        y=alt.Y(f'Component{comp2+1}:Q', title=f'Component {comp2+1}'),
        color=alt.Color('OnesDigit:N', title='Ones Digit',
                      scale=alt.Scale(scheme='category10')),
        tooltip=['Number', f'Component{comp1+1}', f'Component{comp2+1}', 'OnesDigit']
    ).properties(
        title=f'{embeddings_data}: Correlation Between Components {comp1+1} and {comp2+1}',
        width=600,
        height=600
    ).interactive()
    
    return chart

def visualize_digit_patterns(embeddings_data: EmbeddingsData, n_components: int = 5):
    """Generate a set of visualizations highlighting digit patterns in the embeddings."""
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(embeddings_data.data)
    
    # Create visualizations
    visualizations = {
        'component2_with_numbers': plot_component_with_numbers(embeddings_data, pca, 2),
        'component4_with_numbers': plot_component_with_numbers(embeddings_data, pca, 4),
        'digit_periodicity_comp2': plot_digit_periodicity(embeddings_data, pca, 2),
        'digit_periodicity_comp4': plot_digit_periodicity(embeddings_data, pca, 4),
        'digit_value_comp2': plot_digit_value_comparison(embeddings_data, pca, 2),
        'digit_value_comp4': plot_digit_value_comparison(embeddings_data, pca, 4),
        'heatmap_periodicity': plot_heatmap_periodicity(embeddings_data, pca, [2, 4]),
        'radar_digits': plot_radar_digits(embeddings_data, pca),
        'component_correlation': plot_component_correlation(embeddings_data, pca, 2, 4)
    }
    
    return visualizations

# Example usage:
# Assuming 'embeddings' is an instance of EmbeddingsData
# viz = visualize_digit_patterns(embeddings)
# viz['digit_periodicity_comp2']  # Display specific visualization