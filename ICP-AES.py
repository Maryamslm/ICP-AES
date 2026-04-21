import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido
from scipy import stats

# Page configuration
st.set_page_config(page_title="ICP-AES Publication Graphs", layout="wide", page_icon="📊")
st.title("🔬 ICP-AES Results: Advanced Visualization Suite")
st.markdown("### Cobalt-Chromium Alloys in Ringer's & Lactic Acid Solutions")

# ============================================
# DATA ENTRY (from your DOCX file)
# ============================================

def create_ringer_7d():
    data = {
        'Sample': ['CH0-Ringer-7D', 'CH45-Ringer-7D', 'PH0-Ringer-7D', 'PH45-Ringer-7D',
                   'CNH0-Ringer-7D', 'CNH45-Ringer-7D', 'PNH0-Ringer-7D', 'PNH45-Ringer-7D'],
        'Co': [0.099, 0.112, 0.088, 0.095, 0.079, 0.082, 0.069, 0.071],
        'Co_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Cr': [0.068, 0.088, 0.077, 0.086, 0.044, 0.055, 0.064, 0.071],
        'Cr_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'W': [0.044, 0.055, 0.051, 0.055, 0.022, 0.031, 0.029, 0.033],
        'W_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Mo': [0.022, 0.031, 0.025, 0.035, 0.011, 0.020, 0.015, 0.025],
        'Mo_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }
    return pd.DataFrame(data)

def create_ringer_1m():
    data = {
        'Sample': ['CH0-Ringer-1M', 'CH45-Ringer-1M', 'PH0-Ringer-1M', 'PH45-Ringer-1M',
                   'CNH0-Ringer-1M', 'CNH45-Ringer-1M', 'PNH0-Ringer-1M', 'PNH45-Ringer-1M'],
        'Co': [0.145, 0.167, 0.175, 0.191, 0.111, 0.122, 0.133, 0.145],
        'Co_err': [0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001],
        'Cr': [0.088, 0.098, 0.099, 0.101, 0.068, 0.088, 0.078, 0.085],
        'Cr_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Si': [0.012, 0.019, 0.020, 0.021, 0.009, 0.010, 0.010, 0.020],
        'Si_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'W': [0.066, 0.071, 0.078, 0.081, 0.055, 0.059, 0.061, 0.065],
        'W_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Mo': [0.044, 0.050, 0.055, 0.059, 0.022, 0.029, 0.035, 0.040],
        'Mo_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Al': [0.009, 0.010, 0.010, 0.015, 0, 0, 0, 0],
        'Al_err': [0.001, 0.001, 0.001, 0.001, 0, 0, 0, 0],
    }
    return pd.DataFrame(data)

def create_lac_7d():
    data = {
        'Sample': ['CH0-LAC-7D', 'CH45-LAC-7D', 'PH0-LAC-7D', 'PH45-LAC-7D',
                   'CNH0-LAC-7D', 'CNH45-LAC-7D', 'PNH0-LAC-7D', 'PNH45-LAC-7D'],
        'Co': [1.45, 1.77, 1.89, 1.99, 1.22, 1.34, 1.33, 1.55],
        'Co_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Cr': [0.089, 0.095, 0.099, 0.110, 0.074, 0.080, 0.077, 0.085],
        'Cr_err': [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
        'Si': [0.034, 0.038, 0.044, 0.051, 0.022, 0.029, 0.029, 0.031],
        'Si_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'W': [0.069, 0.074, 0.086, 0.088, 0.044, 0.048, 0.051, 0.055],
        'W_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Mo': [0.046, 0.049, 0.051, 0.054, 0.022, 0.029, 0.033, 0.041],
        'Mo_err': [0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001],
    }
    return pd.DataFrame(data)

def create_lac_1m():
    data = {
        'Sample': ['CH0-LAC-1M', 'CH45-LAC-1M', 'PH0-LAC-1M', 'PH45-LAC-1M',
                   'CNH0-LAC-1M', 'CNH45-LAC-1M', 'PNH0-LAC-1M', 'PNH45-LAC-1M'],
        'Co': [1.91, 1.99, 2.11, 2.34, 1.11, 1.33, 1.35, 1.55],
        'Co_err': [0.01, 0.01, 0.03, 0.03, 0.02, 0.01, 0.02, 0.01],
        'Cr': [0.101, 0.111, 0.211, 0.232, 0.094, 0.098, 0.067, 0.071],
        'Cr_err': [0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Si': [0.055, 0.069, 0.065, 0.072, 0.049, 0.055, 0.042, 0.049],
        'Si_err': [0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001],
        'W': [0.077, 0.085, 0.085, 0.091, 0.055, 0.060, 0.057, 0.065],
        'W_err': [0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Mo': [0.038, 0.045, 0.049, 0.055, 0.025, 0.030, 0.022, 0.031],
        'Mo_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Al': [0.022, 0.025, 0.029, 0.031, 0.012, 0.015, 0.022, 0.028],
        'Al_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Fe': [0.011, 0.014, 0.017, 0.020, 0.009, 0.010, 0.013, 0.019],
        'Fe_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }
    return pd.DataFrame(data)

# ============================================
# MODERN PLOTTING FUNCTIONS
# ============================================

def set_font_sizes(fig, base_font_size=12):
    """Apply consistent font sizes to matplotlib figure"""
    plt.rcParams['font.size'] = base_font_size
    plt.rcParams['axes.titlesize'] = base_font_size + 2
    plt.rcParams['axes.labelsize'] = base_font_size
    plt.rcParams['xtick.labelsize'] = base_font_size - 1
    plt.rcParams['ytick.labelsize'] = base_font_size - 1
    plt.rcParams['legend.fontsize'] = base_font_size - 1
    return fig

def download_figure_matplotlib(fig, filename):
    """Convert matplotlib figure to downloadable PNG"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
    return href

def download_figure_plotly(fig, filename):
    """Convert plotly figure to downloadable PNG"""
    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
    return href

# ============================================
# MATPLOTLIB VISUALIZATIONS
# ============================================

def plot_grouped_bars_matplotlib(df, elements, title, ylabel, font_size=12, color_palette='viridis'):
    """Create grouped bar plot with error bars"""
    samples = df['Sample'].tolist()
    x = np.arange(len(samples))
    width = 0.8 / len(elements)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette(color_palette, len(elements))
    
    for i, element in enumerate(elements):
        if element in df.columns:
            values = df[element].values
            errors = df[f'{element}_err'].values if f'{element}_err' in df.columns else np.zeros_like(values)
            offset = (i - len(elements)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=element, 
                          color=colors[i], edgecolor='black', linewidth=0.5)
            ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                        ecolor='black', capsize=3, capthick=1)
    
    ax.set_xlabel('Sample', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=font_size-1)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=font_size-1)
    ax.set_title(title, fontsize=font_size+2, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

def plot_scatter_plot(df, x_element, y_element, font_size=12):
    """Create publication-ready scatter plot with trend line"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract groups
    groups = df['Sample'].str.extract(r'(CH|PH|CNH|PNH)')[0]
    unique_groups = groups.unique()
    colors = sns.color_palette("husl", len(unique_groups))
    
    for group, color in zip(unique_groups, colors):
        mask = groups == group
        x_vals = df[mask][x_element].values
        y_vals = df[mask][y_element].values
        ax.scatter(x_vals, y_vals, label=group, s=100, alpha=0.7, 
                  color=color, edgecolors='black', linewidth=1.5)
        
        # Add trend line
        if len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), '--', color=color, alpha=0.5, linewidth=1)
    
    ax.set_xlabel(f'{x_element} Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(f'{y_element} Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_title(f'Correlation: {x_element} vs {y_element}', fontsize=font_size+2, fontweight='bold')
    ax.legend(frameon=True, fontsize=font_size-1, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add correlation coefficient
    corr = df[x_element].corr(df[y_element])
    ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax.transAxes, 
            fontsize=font_size, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

def plot_radar_chart(df, sample_names, elements, font_size=12):
    """Create radar chart for multi-element comparison"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(elements), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = sns.color_palette("tab10", len(sample_names))
    
    for idx, sample in enumerate(sample_names):
        if sample in df['Sample'].values:
            values = df[df['Sample'] == sample][elements].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=sample, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(elements, fontsize=font_size-1)
    ax.set_ylim(0, max(df[elements].max().max() * 1.1))
    ax.set_title('Element Profile Radar Chart', fontsize=font_size+2, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=font_size-1)
    ax.grid(True)
    
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

def plot_violin_distribution(df, elements, font_size=12):
    """Create violin plot showing data distribution"""
    # Melt dataframe for seaborn
    df_melted = df.melt(id_vars=['Sample'], value_vars=elements, 
                        var_name='Element', value_name='Concentration')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(data=df_melted, x='Element', y='Concentration', 
                   palette='Set3', ax=ax, cut=0)
    sns.swarmplot(data=df_melted, x='Element', y='Concentration', 
                  color='black', alpha=0.6, size=4, ax=ax)
    
    ax.set_xlabel('Element', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_title('Distribution of Element Concentrations', fontsize=font_size+2, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

# ============================================
# PLOTLY INTERACTIVE VISUALIZATIONS
# ============================================

def plot_3d_scatter(df, x_element, y_element, z_element, font_size=12):
    """Create interactive 3D scatter plot"""
    fig = px.scatter_3d(df, x=x_element, y=y_element, z=z_element,
                        color='Sample', size_max=10,
                        title=f'3D Visualization: {x_element} vs {y_element} vs {z_element}',
                        labels={x_element: f'{x_element} (mg/L)',
                               y_element: f'{y_element} (mg/L)',
                               z_element: f'{z_element} (mg/L)'})
    
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     legend_title_font_size=font_size)
    
    return fig

def plot_heatmap(df, elements, font_size=12):
    """Create correlation heatmap"""
    corr_matrix = df[elements].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                   color_continuous_scale='RdBu_r',
                   title='Element Correlation Heatmap',
                   labels=dict(color='Correlation'))
    
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     width=800, height=700)
    
    return fig

def plot_sunburst(df, elements):
    """Create sunburst chart for composition analysis"""
    # Prepare hierarchical data
    avg_values = df[elements].mean()
    
    # Create sunburst data
    fig = go.Figure(go.Sunburst(
        labels=['Total'] + elements,
        parents=[''] + ['Total'] * len(elements),
        values=[sum(avg_values)] + avg_values.tolist(),
        branchvalues="total",
        marker=dict(colors=px.colors.qualitative.Set3),
        textinfo="label+percent entry"
    ))
    
    fig.update_layout(title='Element Composition Sunburst Chart',
                     width=800, height=800)
    
    return fig

def plot_parallel_coordinates(df, elements, font_size=12):
    """Create parallel coordinates plot for multivariate analysis"""
    fig = px.parallel_coordinates(df, dimensions=elements,
                                  color='Co' if 'Co' in df.columns else elements[0],
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  title='Parallel Coordinates: Multi-element Comparison')
    
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     width=1000, height=600)
    
    return fig

def plot_bubble_chart(df, x_element, y_element, size_element, font_size=12):
    """Create bubble chart with third dimension"""
    fig = px.scatter(df, x=x_element, y=y_element, size=size_element,
                    color='Sample', hover_name='Sample',
                    size_max=30,
                    title=f'Bubble Chart: {x_element} vs {y_element} (size: {size_element})',
                    labels={x_element: f'{x_element} (mg/L)',
                           y_element: f'{y_element} (mg/L)'})
    
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     showlegend=True)
    
    return fig

def plot_pie_chart(df, element, timepoint, font_size=12):
    """Create pie chart for element distribution across samples"""
    values = df[element].values
    labels = df['Sample'].values
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                                 hole=0.3,
                                 textinfo='label+percent',
                                 textposition='auto')])
    
    fig.update_layout(title=f'{element} Distribution - {timepoint}',
                     font=dict(size=font_size),
                     title_font_size=font_size+2,
                     width=800, height=600)
    
    return fig

def plot_donut_comparison(df_ringer, df_lac, element, font_size=12):
    """Compare element distribution between solutions using donut charts"""
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['Ringer\'s Solution', 'Lactic Acid + NaCl'])
    
    # Ringer's data
    values_ringer = df_ringer[element].values
    labels = df_ringer['Sample'].values
    
    # Lactic acid data
    values_lac = df_lac[element].values
    
    fig.add_trace(go.Pie(labels=labels, values=values_ringer, hole=0.4,
                        name='Ringer\'s', marker=dict(colors=px.colors.qualitative.Set3)),
                  row=1, col=1)
    
    fig.add_trace(go.Pie(labels=labels, values=values_lac, hole=0.4,
                        name='Lactic Acid', marker=dict(colors=px.colors.qualitative.Set3)),
                  row=1, col=2)
    
    fig.update_layout(title_text=f'{element} Distribution Comparison',
                     font=dict(size=font_size),
                     title_font_size=font_size+2,
                     width=1200, height=600)
    
    return fig

def plot_solution_comparison(df_ringer, df_lac, element, font_size=12):
    """Compare Ringer vs Lactic Acid solutions"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df_ringer['Sample']))
    width = 0.35
    
    vals_ringer = df_ringer[element].values
    vals_lac = df_lac[element].values
    
    bars1 = ax.bar(x - width/2, vals_ringer, width, label='Ringer\'s Solution',
                   color='#4682B4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, vals_lac, width, label='Lactic Acid + NaCl',
                   color='#D55E00', edgecolor='black', linewidth=0.5)
    
    # Add error bars if available
    if f'{element}_err' in df_ringer.columns:
        ax.errorbar(x - width/2, vals_ringer, yerr=df_ringer[f'{element}_err'].values,
                   fmt='none', ecolor='black', capsize=3)
    if f'{element}_err' in df_lac.columns:
        ax.errorbar(x + width/2, vals_lac, yerr=df_lac[f'{element}_err'].values,
                   fmt='none', ecolor='black', capsize=3)
    
    ax.set_xlabel('Sample', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(f'{element} Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_ringer['Sample'], rotation=45, ha='right', fontsize=font_size-1)
    ax.legend(frameon=False, fontsize=font_size, loc='upper left')
    ax.set_title(f'{element} Release: Ringer\'s vs Lactic Acid Solution', 
                fontsize=font_size+2, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

# ============================================
# MAIN APP INTERFACE
# ============================================

# Sidebar - Global controls
st.sidebar.header("🎛️ Global Settings")
font_size = st.sidebar.slider("Font Size (pts)", 8, 20, 12, help="Base font size for all plots")
color_palette = st.sidebar.selectbox("Color Palette", ["viridis", "plasma", "Set2", "tab10", "Set1", "husl"])
plot_engine = st.sidebar.selectbox("Plot Engine", ["Matplotlib (Static)", "Plotly (Interactive)"], 
                                   help="Matplotlib for publication, Plotly for exploration")

# Data selection
st.sidebar.markdown("---")
st.sidebar.header("📊 Data Selection")
timepoint = st.sidebar.selectbox("Time Point", ["7 days", "1 month"])
solution_type = st.sidebar.multiselect("Solutions to Compare", 
                                        ["Ringer's Solution", "Lactic Acid + NaCl"],
                                        default=["Ringer's Solution"])

# Load appropriate data
if "Ringer's Solution" in solution_type:
    df_ringer_7d = create_ringer_7d()
    df_ringer_1m = create_ringer_1m()
if "Lactic Acid + NaCl" in solution_type:
    df_lac_7d = create_lac_7d()
    df_lac_1m = create_lac_1m()

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Bar & Comparison", 
    "🔬 Advanced Analytics", 
    "🎨 Modern Visualizations",
    "📈 Interactive Plotly",
    "⚖️ Solution Comparison"
])

# ============================================
# TAB 1: BAR & COMPARISON PLOTS
# ============================================
with tab1:
    st.header("Classic Bar Charts & Comparisons")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "Ringer's Solution" in solution_type:
            df_current_ringer = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
            available_elements_ringer = [col for col in df_current_ringer.columns if col not in ['Sample', 'Sample_clean'] and not col.endswith('_err')]
        if "Lactic Acid + NaCl" in solution_type:
            df_current_lac = df_lac_7d if timepoint == "7 days" else df_lac_1m
            available_elements_lac = [col for col in df_current_lac.columns if col not in ['Sample', 'Sample_clean'] and not col.endswith('_err')]
        
        available_elements = list(set(available_elements_ringer if "Ringer's Solution" in solution_type else []) | 
                                  set(available_elements_lac if "Lactic Acid + NaCl" in solution_type else []))
        
        selected_elements = st.multiselect("Select Elements", available_elements, default=available_elements[:3])
    
    with col2:
        plot_type = st.radio("Plot Type", ["Grouped Bars", "Scatter Plot", "Radar Chart", "Violin Plot"], horizontal=True)
    
    if selected_elements:
        if plot_type == "Grouped Bars":
            for sol in solution_type:
                if sol == "Ringer's Solution":
                    df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                    fig = plot_grouped_bars_matplotlib(df_current, selected_elements, 
                                                       f"{sol} - {timepoint}", "Concentration (mg/L)",
                                                       font_size, color_palette)
                    st.pyplot(fig)
                    if st.button(f"Download {sol} Plot", key=f"download_bar_{sol}"):
                        st.markdown(download_figure_matplotlib(fig, f"{sol}_{timepoint}_bars"), unsafe_allow_html=True)
        
        elif plot_type == "Scatter Plot" and len(selected_elements) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_element = st.selectbox("X-axis Element", selected_elements, key="scatter_x")
            with col2:
                y_element = st.selectbox("Y-axis Element", selected_elements, key="scatter_y")
            
            for sol in solution_type:
                if sol == "Ringer's Solution":
                    df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                    fig = plot_scatter_plot(df_current, x_element, y_element, font_size)
                    st.pyplot(fig)
                    if st.button(f"Download {sol} Scatter", key=f"download_scatter_{sol}"):
                        st.markdown(download_figure_matplotlib(fig, f"{sol}_{timepoint}_scatter"), unsafe_allow_html=True)
        
        elif plot_type == "Radar Chart":
            samples_to_plot = st.multiselect("Select Samples for Radar Chart", 
                                            df_current_ringer['Sample'].tolist() if "Ringer's Solution" in solution_type else df_current_lac['Sample'].tolist(),
                                            default=df_current_ringer['Sample'].tolist()[:4] if "Ringer's Solution" in solution_type else df_current_lac['Sample'].tolist()[:4])
            
            if samples_to_plot:
                for sol in solution_type:
                    if sol == "Ringer's Solution":
                        df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                        fig = plot_radar_chart(df_current, samples_to_plot, selected_elements, font_size)
                        st.pyplot(fig)
        
        elif plot_type == "Violin Plot":
            for sol in solution_type:
                if sol == "Ringer's Solution":
                    df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                    fig = plot_violin_distribution(df_current, selected_elements, font_size)
                    st.pyplot(fig)

# ============================================
# TAB 2: ADVANCED ANALYTICS
# ============================================
with tab2:
    st.header("🔬 Statistical & Distribution Analysis")
    
    if len(solution_type) > 0:
        df_current = df_ringer_7d if timepoint == "7 days" and "Ringer's Solution" in solution_type else df_ringer_1m if "Ringer's Solution" in solution_type else df_lac_7d if "Lactic Acid + NaCl" in solution_type else df_lac_1m
        
        elements_avail = [col for col in df_current.columns if col not in ['Sample', 'Sample_clean'] and not col.endswith('_err')]
        
        analysis_type = st.selectbox("Analysis Type", 
                                    ["Correlation Matrix", "Box Plots", "Statistical Summary", "ANOVA Test"])
        
        if analysis_type == "Correlation Matrix":
            selected_corr = st.multiselect("Select Elements for Correlation", elements_avail, default=elements_avail[:4])
            if len(selected_corr) >= 2:
                corr_matrix = df_current[selected_corr].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title('Element Correlation Matrix', fontsize=font_size+2, fontweight='bold')
                st.pyplot(fig)
        
        elif analysis_type == "Box Plots":
            df_melted = df_current.melt(id_vars=['Sample'], value_vars=selected_corr if 'selected_corr' in locals() else elements_avail[:4],
                                       var_name='Element', value_name='Concentration')
            fig, ax = plt.subplots(figsize=(12, 6))
            df_melted.boxplot(column='Concentration', by='Element', ax=ax)
            ax.set_title('Element Concentration Distribution', fontsize=font_size+2, fontweight='bold')
            ax.set_xlabel('Element', fontsize=font_size)
            ax.set_ylabel('Concentration (mg/L)', fontsize=font_size)
            plt.suptitle('')
            st.pyplot(fig)
        
        elif analysis_type == "Statistical Summary":
            st.dataframe(df_current[elements_avail].describe().round(4))
            
            # Add coefficient of variation
            cv = df_current[elements_avail].std() / df_current[elements_avail].mean() * 100
            st.write("**Coefficient of Variation (%)**")
            st.dataframe(cv.round(2))

# ============================================
#
