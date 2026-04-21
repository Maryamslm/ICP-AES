import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Handle Plotly imports gracefully
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Handle Kaleido imports gracefully
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# Handle SciPy imports gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="ICP-AES Publication Graphs", layout="wide", page_icon="📊")
st.title("🔬 ICP-AES Results: Advanced Visualization Suite")
st.markdown("### Cobalt-Chromium Alloys in Ringer's & Lactic Acid Solutions")

# ============================================
# DATA ENTRY FUNCTIONS
# ============================================

def create_ringer_7d():
    data = {
        'Sample': ['CH0-Ringer-7D', 'CH45-Ringer-7D', 'PH0-Ringer-7D', 'PH45-Ringer-7D',
                   'CNH0-Ringer-7D', 'CNH45-Ringer-7D', 'PNH0-Ringer-7D', 'PNH45-Ringer-7D'],
        'Co': [0.099, 0.112, 0.088, 0.095, 0.079, 0.082, 0.069, 0.071],
        'Co_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Cr': [0.068, 0.088, 0.077, 0.086, 0.044, 0.055, 0.064, 0.071],
        'Cr_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Si': [0.008, 0.011, 0.013, 0.015, 0.006, 0.008, 0.009, 0.012],  # ✅ Si added
        'Si_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
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
# HELPER FUNCTIONS
# ============================================

def set_font_sizes(fig, base_font_size=12):
    """Apply consistent font sizes to matplotlib figure"""
    plt.rcParams.update({
        'font.size': base_font_size,
        'axes.titlesize': base_font_size + 2,
        'axes.labelsize': base_font_size,
        'xtick.labelsize': base_font_size - 1,
        'ytick.labelsize': base_font_size - 1,
        'legend.fontsize': base_font_size - 1
    })
    return fig

def download_figure_matplotlib(fig, filename):
    """Convert matplotlib figure to downloadable PNG"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
    return href

def download_figure_plotly(fig, filename):
    """Convert plotly figure to downloadable PNG"""
    if KALEIDO_AVAILABLE:
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
        return href
    else:
        return "⚠️ `kaleido` not installed. Run `pip install kaleido`"

# ============================================
# MATPLOTLIB VISUALIZATIONS
# ============================================

def plot_grouped_bars_matplotlib(df, elements, title, ylabel, font_size=12, color_palette='viridis'):
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
            ax.bar(x + offset, values, width, label=element, 
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
    fig, ax = plt.subplots(figsize=(10, 8))
    groups = df['Sample'].str.extract(r'(CH|PH|CNH|PNH)')[0]
    unique_groups = groups.unique()
    colors = sns.color_palette("husl", len(unique_groups))
    
    for group, color in zip(unique_groups, colors):
        mask = groups == group
        x_vals = df[mask][x_element].values
        y_vals = df[mask][y_element].values
        ax.scatter(x_vals, y_vals, label=group, s=100, alpha=0.7, 
                   color=color, edgecolors='black', linewidth=1.5)
        if len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), '--', color=color, alpha=0.5, linewidth=1)
    
    ax.set_xlabel(f'{x_element} Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(f'{y_element} Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_title(f'Correlation: {x_element} vs {y_element}', fontsize=font_size+2, fontweight='bold')
    ax.legend(frameon=True, fontsize=font_size-1, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    corr = df[x_element].corr(df[y_element])
    ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax.transAxes, 
            fontsize=font_size, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

def plot_radar_chart(df, sample_names, elements, font_size=12):
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
    max_val = df[elements].max().max() * 1.2
    ax.set_ylim(0, max_val)
    ax.set_title('Element Profile Radar Chart', fontsize=font_size+2, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=font_size-1)
    ax.grid(True)
    plt.tight_layout()
    return set_font_sizes(fig, font_size)

def plot_violin_distribution(df, elements, font_size=12):
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
    fig = px.scatter_3d(df, x=x_element, y=y_element, z=z_element,
                        color='Sample', size_max=10,
                        title=f'3D Visualization: {x_element} vs {y_element} vs {z_element}')
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     legend_title_font_size=font_size)
    return fig

def plot_heatmap(df, elements, font_size=12):
    corr_matrix = df[elements].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Element Correlation Heatmap')
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     width=800, height=700)
    return fig

def plot_sunburst(df, elements):
    avg_values = df[elements].mean()
    fig = go.Figure(go.Sunburst(
        labels=['Total'] + elements,
        parents=[''] + ['Total'] * len(elements),
        values=[sum(avg_values)] + avg_values.tolist(),
        branchvalues="total",
        marker=dict(colors=px.colors.qualitative.Set3),
        textinfo="label+percent entry"
    ))
    fig.update_layout(title='Element Composition Sunburst Chart', width=800, height=800)
    return fig

def plot_parallel_coordinates(df, elements, font_size=12):
    fig = px.parallel_coordinates(df, dimensions=elements,
                                  color='Co' if 'Co' in df.columns else elements[0],
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  title='Parallel Coordinates: Multi-element Comparison')
    fig.update_layout(font=dict(size=font_size),
                     title_font_size=font_size+2,
                     width=1000, height=600)
    return fig

def plot_bubble_chart(df, x_element, y_element, size_element, font_size=12):
    fig = px.scatter(df, x=x_element, y=y_element, size=size_element,
                     color='Sample', hover_name='Sample', size_max=30,
                     title=f'Bubble Chart: {x_element} vs {y_element} (size: {size_element})')
    fig.update_layout(font=dict(size=font_size), title_font_size=font_size+2)
    return fig

def plot_donut_comparison(df_ringer, df_lac, element, font_size=12):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=["Ringer's Solution", "Lactic Acid + NaCl"])
    values_ringer = df_ringer[element].values
    values_lac = df_lac[element].values
    labels = df_ringer['Sample'].values
    
    fig.add_trace(go.Pie(labels=labels, values=values_ringer, hole=0.4, name="Ringer's",
                         marker=dict(colors=px.colors.qualitative.Set3)), row=1, col=1)
    fig.add_trace(go.Pie(labels=labels, values=values_lac, hole=0.4, name="Lactic Acid",
                         marker=dict(colors=px.colors.qualitative.Set3)), row=1, col=2)
    fig.update_layout(title_text=f'{element} Distribution Comparison',
                     font=dict(size=font_size), title_font_size=font_size+2,
                     width=1200, height=600)
    return fig

def plot_solution_comparison(df_ringer, df_lac, element, font_size=12):
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df_ringer['Sample']))
    width = 0.35
    vals_ringer = df_ringer[element].values
    vals_lac = df_lac[element].values
    
    ax.bar(x - width/2, vals_ringer, width, label="Ringer's Solution",
           color='#4682B4', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, vals_lac, width, label="Lactic Acid + NaCl",
           color='#D55E00', edgecolor='black', linewidth=0.5)
    
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

st.sidebar.header("🎛️ Global Settings")
font_size = st.sidebar.slider("Font Size (pts)", 8, 20, 12, help="Base font size for all plots")
color_palette = st.sidebar.selectbox("Color Palette", ["viridis", "plasma", "Set2", "tab10", "Set1", "husl"])
plot_engine = st.sidebar.selectbox("Plot Engine", ["Matplotlib (Static)", "Plotly (Interactive)"], 
                                   help="Matplotlib for publication, Plotly for exploration")

st.sidebar.markdown("---")
st.sidebar.header("📊 Data Selection")
timepoint = st.sidebar.selectbox("Time Point", ["7 days", "1 month"])
solution_type = st.sidebar.multiselect("Solutions to Compare", 
                                        ["Ringer's Solution", "Lactic Acid + NaCl"],
                                        default=["Ringer's Solution"])

# Load data
df_ringer_7d = create_ringer_7d()
df_ringer_1m = create_ringer_1m()
df_lac_7d = create_lac_7d()
df_lac_1m = create_lac_1m()

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
        available_elements = set()
        if "Ringer's Solution" in solution_type:
            df_temp = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
            available_elements |= set([c for c in df_temp.columns if not c.endswith('_err') and c != 'Sample'])
        if "Lactic Acid + NaCl" in solution_type:
            df_temp = df_lac_7d if timepoint == "7 days" else df_lac_1m
            available_elements |= set([c for c in df_temp.columns if not c.endswith('_err') and c != 'Sample'])
        available_elements = sorted(list(available_elements))
        selected_elements = st.multiselect("Select Elements", available_elements, default=available_elements[:3])
    
    with col2:
        plot_type = st.radio("Plot Type", ["Grouped Bars", "Scatter Plot", "Radar Chart", "Violin Plot"], horizontal=True)
    
    if selected_elements:
        if plot_type == "Grouped Bars":
            for sol in solution_type:
                if sol == "Ringer's Solution":
                    df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                else:
                    df_current = df_lac_7d if timepoint == "7 days" else df_lac_1m
                fig = plot_grouped_bars_matplotlib(df_current, selected_elements, 
                                                   f"{sol} - {timepoint}", "Concentration (mg/L)",
                                                   font_size, color_palette)
                st.pyplot(fig)
                if st.button(f"Download {sol} Bar Plot", key=f"dl_bar_{sol}"):
                    st.markdown(download_figure_matplotlib(fig, f"{sol}_{timepoint}_bars"), unsafe_allow_html=True)
        
        elif plot_type == "Scatter Plot" and len(selected_elements) >= 2:
            cx, cy = st.columns(2)
            with cx: x_el = st.selectbox("X-axis", selected_elements, key="sx")
            with cy: y_el = st.selectbox("Y-axis", selected_elements, key="sy")
            
            for sol in solution_type:
                if sol == "Ringer's Solution":
                    df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                else:
                    df_current = df_lac_7d if timepoint == "7 days" else df_lac_1m
                fig = plot_scatter_plot(df_current, x_el, y_el, font_size)
                st.pyplot(fig)
                if st.button(f"Download {sol} Scatter", key=f"dl_sc_{sol}"):
                    st.markdown(download_figure_matplotlib(fig, f"{sol}_{timepoint}_scatter"), unsafe_allow_html=True)
        
        elif plot_type == "Radar Chart":
            df_radar = df_ringer_7d if timepoint == "7 days" and "Ringer's Solution" in solution_type else df_lac_7d if "Lactic Acid + NaCl" in solution_type else df_ringer_1m
            samples_avail = df_radar['Sample'].tolist()
            samples_plot = st.multiselect("Samples", samples_avail, default=samples_avail[:3])
            if samples_plot:
                fig = plot_radar_chart(df_radar, samples_plot, selected_elements, font_size)
                st.pyplot(fig)
        
        elif plot_type == "Violin Plot":
            for sol in solution_type:
                if sol == "Ringer's Solution":
                    df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
                else:
                    df_current = df_lac_7d if timepoint == "7 days" else df_lac_1m
                fig = plot_violin_distribution(df_current, selected_elements, font_size)
                st.pyplot(fig)

# ============================================
# TAB 2: ADVANCED ANALYTICS
# ============================================
with tab2:
    st.header("🔬 Statistical & Distribution Analysis")
    
    if len(solution_type) > 0:
        if "Ringer's Solution" in solution_type:
            df_current = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
        else:
            df_current = df_lac_7d if timepoint == "7 days" else df_lac_1m
        
        elements_avail = [c for c in df_current.columns if not c.endswith('_err') and c != 'Sample']
        analysis_type = st.selectbox("Analysis Type", ["Correlation Matrix", "Box Plots", "Statistical Summary", "ANOVA Test"])
        
        if analysis_type == "Correlation Matrix":
            sel_corr = st.multiselect("Elements", elements_avail, default=elements_avail[:4])
            if len(sel_corr) >= 2:
                corr = df_current[sel_corr].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, 
                           linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title('Element Correlation Matrix', fontsize=font_size+2, fontweight='bold')
                st.pyplot(fig)
        
        elif analysis_type == "Box Plots":
            df_m = df_current.melt(id_vars=['Sample'], value_vars=elements_avail[:5],
                                   var_name='Element', value_name='Concentration')
            fig, ax = plt.subplots(figsize=(12, 6))
            df_m.boxplot(column='Concentration', by='Element', ax=ax)
            ax.set_title('Concentration Distribution', fontsize=font_size+2, fontweight='bold')
            ax.set_xlabel('Element', fontsize=font_size)
            ax.set_ylabel('Concentration (mg/L)', fontsize=font_size)
            plt.suptitle('')
            st.pyplot(fig)
        
        elif analysis_type == "Statistical Summary":
            st.dataframe(df_current[elements_avail].describe().round(4))
            cv = df_current[elements_avail].std() / df_current[elements_avail].mean() * 100
            st.subheader("Coefficient of Variation (%)")
            st.dataframe(cv.round(2))
        
        elif analysis_type == "ANOVA Test":
            st.info("Statistical significance testing across sample groups")
            if SCIPY_AVAILABLE:
                st.write("ANOVA requires grouping samples (e.g., CH vs PH vs CNH vs PNH).")
            else:
                st.warning("scipy not installed")

# ============================================
# TAB 3: MODERN VISUALIZATIONS
# ============================================
with tab3:
    st.header("🎨 Modern Visualizations")
    
    if "Ringer's Solution" in solution_type:
        df_mod = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
    else:
        df_mod = df_lac_7d if timepoint == "7 days" else df_lac_1m
    
    elements_mod = [c for c in df_mod.columns if not c.endswith('_err') and c != 'Sample']
    
    viz_type = st.selectbox("Visualization Type", ["Stacked Bar", "Line Chart", "Element Stacking %", "Pairplot"])
    
    if viz_type == "Stacked Bar":
        sel_stacked = st.multiselect("Elements to Stack", elements_mod, default=elements_mod[:3])
        if sel_stacked:
            fig, ax = plt.subplots(figsize=(12, 7))
            df_plot = df_mod.set_index('Sample')[sel_stacked]
            df_plot.plot(kind='bar', stacked=True, ax=ax, colormap=color_palette)
            ax.set_ylabel('Concentration (mg/L)', fontsize=font_size, fontweight='bold')
            ax.set_title(f'Stacked Bar: {timepoint}', fontsize=font_size+2, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size-1)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(set_font_sizes(fig, font_size))
    
    elif viz_type == "Line Chart":
        sel_line = st.multiselect("Elements", elements_mod, default=['Co', 'Cr'])
        if sel_line:
            fig, ax = plt.subplots(figsize=(12, 6))
            for el in sel_line:
                ax.plot(df_mod['Sample'], df_mod[el], marker='o', label=el, linewidth=2)
            ax.set_ylabel('Concentration (mg/L)', fontsize=font_size, fontweight='bold')
            ax.set_title(f'Trend Lines: {timepoint}', fontsize=font_size+2, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=font_size-1)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(set_font_sizes(fig, font_size))
    
    elif viz_type == "Element Stacking %":
        df_pct = df_mod[elements_mod].div(df_mod[elements_mod].sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(12, 7))
        df_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Pastel1')
        ax.set_ylabel('Relative Composition (%)', fontsize=font_size, fontweight='bold')
        ax.set_title(f'Element Composition (%): {timepoint}', fontsize=font_size+2, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size-1)
        plt.tight_layout()
        st.pyplot(set_font_sizes(fig, font_size))
    
    elif viz_type == "Pairplot":
        sel_pair = st.multiselect("Elements (max 5 for performance)", elements_mod, default=elements_mod[:3])
        if len(sel_pair) >= 2:
            with st.spinner("Generating pairplot..."):
                g = sns.pairplot(df_mod[sel_pair], diag_kind='kde', corner=False, 
                                 plot_kws={'alpha': 0.7}, height=3)
                g.fig.suptitle(f'Pairwise Relationships: {timepoint}', y=1.02, fontsize=font_size+2)
                st.pyplot(g.fig)

# ============================================
# TAB 4: INTERACTIVE PLOTLY
# ============================================
with tab4:
    st.header("📈 Interactive Plotly Visualizations")
    
    if not PLOTLY_AVAILABLE:
        st.error("⚠️ Plotly is not installed. Run `pip install plotly`")
    else:
        df_pl = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if "Ringer's Solution" in solution_type else df_lac_7d if timepoint == "7 days" else df_lac_1m
        elements_pl = [c for c in df_pl.columns if not c.endswith('_err') and c != 'Sample']
        
        plotly_type = st.selectbox("Plotly Chart Type", 
                                   ["3D Scatter", "Heatmap", "Sunburst", "Parallel Coordinates", "Bubble Chart", "Donut Comparison"])
        
        if plotly_type == "3D Scatter" and len(elements_pl) >= 3:
            x3, y3, z3 = st.columns(3)
            with x3: xe = st.selectbox("X", elements_pl, key="px")
            with y3: ye = st.selectbox("Y", elements_pl, key="py")
            with z3: ze = st.selectbox("Z", elements_pl, key="pz")
            fig = plot_3d_scatter(df_pl, xe, ye, ze, font_size)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Heatmap":
            sel_heat = st.multiselect("Elements", elements_pl, default=elements_pl[:5])
            if len(sel_heat) >= 2:
                fig = plot_heatmap(df_pl, sel_heat, font_size)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Sunburst":
            fig = plot_sunburst(df_pl, elements_pl)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Parallel Coordinates":
            fig = plot_parallel_coordinates(df_pl, elements_pl, font_size)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Bubble Chart":
            bx, by, bs = st.columns(3)
            with bx: bxe = st.selectbox("X", elements_pl, key="bx")
            with by: bye = st.selectbox("Y", elements_pl, key="by")
            with bs: bse = st.selectbox("Size", elements_pl, key="bs")
            fig = plot_bubble_chart(df_pl, bxe, bye, bse, font_size)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Donut Comparison" and "Ringer's Solution" in solution_type and "Lactic Acid + NaCl" in solution_type:
            el_donut = st.selectbox("Element", elements_pl)
            df_r_donut = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
            df_l_donut = df_lac_7d if timepoint == "7 days" else df_lac_1m
            fig = plot_donut_comparison(df_r_donut, df_l_donut, el_donut, font_size)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 5: SOLUTION COMPARISON
# ============================================
with tab5:
    st.header("⚖️ Solution Comparison")
    
    if "Ringer's Solution" not in solution_type or "Lactic Acid + NaCl" not in solution_type:
        st.warning("⚠️ Please select both Ringer's Solution and Lactic Acid + NaCl in the sidebar for comparison.")
    else:
        df_r_comp = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
        df_l_comp = df_lac_7d if timepoint == "7 days" else df_lac_1m
        
        common_elements = set([c for c in df_r_comp.columns if not c.endswith('_err') and c != 'Sample']) & \
                          set([c for c in df_l_comp.columns if not c.endswith('_err') and c != 'Sample'])
        common_elements = sorted(list(common_elements))
        
        comp_type = st.selectbox("Comparison Type", ["Side-by-Side Bars", "Fold Change", "Statistical Summary"])
        
        if comp_type == "Side-by-Side Bars":
            el_comp = st.selectbox("Element to Compare", common_elements)
            fig = plot_solution_comparison(df_r_comp, df_l_comp, el_comp, font_size)
            st.pyplot(fig)
            if st.button(f"Download Comparison Plot"):
                st.markdown(download_figure_matplotlib(fig, f"comparison_{el_comp}"), unsafe_allow_html=True)
        
        elif comp_type == "Fold Change":
            st.info("Fold Change = (Lactic Acid) / (Ringer's Solution)")
            fc_data = {}
            for el in common_elements:
                fc = df_l_comp[el].mean() / df_r_comp[el].mean()
                fc_data[el] = fc
            fc_df = pd.DataFrame(list(fc_data.items()), columns=['Element', 'Fold Change'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(fc_df['Element'], fc_df['Fold Change'], 
                          color=fc_df['Fold Change'].apply(lambda x: '#D55E00' if x > 1 else '#4682B4'),
                          edgecolor='black', linewidth=0.5)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='No Change')
            ax.set_ylabel('Fold Change (LAC / Ringer)', fontsize=font_size, fontweight='bold')
            ax.set_title(f'Element Release Fold Change: {timepoint}', fontsize=font_size+2, fontweight='bold')
            ax.legend(fontsize=font_size-1)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, fc_df['Fold Change']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.2f}x', ha='center', va='bottom', fontsize=font_size-1, fontweight='bold')
            plt.tight_layout()
            st.pyplot(set_font_sizes(fig, font_size))
        
        elif comp_type == "Statistical Summary":
            st.subheader("Ringer's Solution")
            st.dataframe(df_r_comp[common_elements].describe().round(4))
            st.subheader("Lactic Acid + NaCl")
            st.dataframe(df_l_comp[common_elements].describe().round(4))
            
            comp_table = pd.DataFrame({
                'Element': common_elements,
                'Ringer Mean': [df_r_comp[e].mean() for e in common_elements],
                'LAC Mean': [df_l_comp[e].mean() for e in common_elements],
                'Difference': [df_l_comp[e].mean() - df_r_comp[e].mean() for e in common_elements],
                'Fold Change': [df_l_comp[e].mean() / df_r_comp[e].mean() for e in common_elements]
            }).round(4)
            st.dataframe(comp_table)

# Footer
st.markdown("---")
st.caption("📊 ICP-AES Visualization Suite | Requires: streamlit, pandas, numpy, matplotlib, seaborn, plotly, kaleido, scipy")
