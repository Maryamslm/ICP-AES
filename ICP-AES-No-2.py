import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# Handle optional dependencies gracefully
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ============================================
# SAFE COLOR RESOLVER (FIXES PALETTE KEYERROR)
# ============================================

def get_safe_colors(palette_name, n_colors):
    """Maps UI palette names to valid seaborn/matplotlib color lists."""
    palette_map = {
        "colorblind": "colorblind",
        "tab10": "tab10",
        "tab20": "tab20",
        "Set1": "Set1",
        "Set2": "Set2",
        "Set3": "Set3",
        "pastel": "pastel",
        "muted": "muted",
        "dark": "dark",
        "bright": "bright",
        "viridis": "viridis",
        "plasma": "plasma",
        "cividis": "cividis",
        "flare": "flare",
        "crest": "crest",
        "mako": "mako",
        "Wong (High-Contrast)": "deep",
        "Publication Standard": "muted"
    }
    safe_name = palette_map.get(palette_name, "viridis")
    try:
        return sns.color_palette(safe_name, n_colors)
    except ValueError:
        return sns.color_palette("viridis", n_colors)

# ============================================
# PAGE CONFIGURATION & STYLING ENGINE
# ============================================

st.set_page_config(page_title="ICP-AES Publication Graphs", layout="wide", page_icon="📊")

def apply_matplotlib_styling(font_family, font_style, font_size, publication_style):
    try:
        plt.rcParams['font.family'] = font_family
    except Exception:
        plt.rcParams['font.family'] = 'sans-serif'
        
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'font.weight': 'bold' if 'Bold' in font_style else 'normal',
        'font.style': 'italic' if 'Italic' in font_style else 'normal'
    })
    
    if publication_style:
        plt.rcParams.update({
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'axes.linewidth': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'legend.frameon': True,
            'legend.facecolor': 'white',
            'legend.edgecolor': 'gray',
            'legend.fancybox': True,
            'legend.framealpha': 0.8,
            'errorbar.capsize': 4
        })
    else:
        plt.rcParams.update({
            'axes.spines.right': True,
            'axes.spines.top': True,
            'axes.grid': False,
            'axes.linewidth': 1.0,
            'xtick.direction': 'out',
            'ytick.direction': 'out'
        })

# ============================================
# DATA ENTRY
# ============================================

def create_ringer_7d():
    return pd.DataFrame({
        'Sample': ['CH0-Ringer-7D', 'CH45-Ringer-7D', 'PH0-Ringer-7D', 'PH45-Ringer-7D',
                   'CNH0-Ringer-7D', 'CNH45-Ringer-7D', 'PNH0-Ringer-7D', 'PNH45-Ringer-7D'],
        'Co': [0.099, 0.112, 0.088, 0.095, 0.079, 0.082, 0.069, 0.071],
        'Co_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Cr': [0.068, 0.088, 0.077, 0.086, 0.044, 0.055, 0.064, 0.071],
        'Cr_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Si': [0.008, 0.011, 0.013, 0.015, 0.006, 0.008, 0.009, 0.012],
        'Si_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'W': [0.044, 0.055, 0.051, 0.055, 0.022, 0.031, 0.029, 0.033],
        'W_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'Mo': [0.022, 0.031, 0.025, 0.035, 0.011, 0.020, 0.015, 0.025],
        'Mo_err': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    })

def create_ringer_1m():
    return pd.DataFrame({
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
    })

def create_lac_7d():
    return pd.DataFrame({
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
    })

def create_lac_1m():
    return pd.DataFrame({
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
    })

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_grouped_data(df, group_by_prefix=False):
    """Return data with optional grouping. Robust dictionary builder prevents column collisions."""
    if not group_by_prefix:
        return df.copy()
        
    df_g = df.copy()
    df_g['Group'] = df_g['Sample'].apply(lambda x: x.split('-')[0])
    
    all_prefixes = ['CH0', 'PH0', 'CNH0', 'PNH0', 'CH45', 'PH45', 'CNH45', 'PNH45']
    valid = [p for p in all_prefixes if p in df_g['Group'].values]
    df_g['Group'] = pd.Categorical(df_g['Group'], categories=valid, ordered=True)
    df_g = df_g.sort_values('Group')
    
    val_cols = [c for c in df_g.columns if not c.endswith('_err') and c not in ['Sample', 'Group']]
    
    result_data = {'Sample': []}
    for c in val_cols:
        result_data[c] = []
        result_data[f'{c}_err'] = []
        
    for group_name, group_df in df_g.groupby('Group'):
        result_data['Sample'].append(group_name)
        for c in val_cols:
            vals = group_df[c].dropna()
            result_data[c].append(vals.mean() if len(vals) > 0 else 0.0)
            err_col = f'{c}_err'
            if err_col in group_df.columns:
                errs = group_df[err_col].dropna()
                if len(errs) > 1:
                    result_data[err_col].append(errs.sem())
                elif len(errs) == 1:
                    result_data[err_col].append(errs.values[0])
                else:
                    result_data[err_col].append(0.0)
            else:
                result_data[err_col].append(0.0)
                
    return pd.DataFrame(result_data)

def download_figure_matplotlib(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
    return href

def download_figure_plotly(fig, filename):
    if KALEIDO_AVAILABLE:
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
        return href
    return "⚠️ `kaleido` not installed for Plotly export"

# ============================================
# MATPLOTLIB VISUALIZATIONS
# ============================================

def plot_grouped_bars_matplotlib(df, elements, title, ylabel, font_size, color_palette, group_by=False):
    df = get_grouped_data(df, group_by)
    samples = df['Sample'].tolist()
    x = np.arange(len(samples))
    width = 0.8 / len(elements) if len(elements) > 0 else 0.8
    
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = get_safe_colors(color_palette, len(elements))
    
    for i, element in enumerate(elements):
        if element in df.columns:
            values = np.asarray(df[element].values).ravel()
            err_col = f'{element}_err'
            errors = np.asarray(df[err_col].values).ravel() if err_col in df.columns else np.zeros_like(values)
            
            if len(values) != len(x):
                continue
                
            offset = (i - len(elements)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=element, 
                   color=colors[i], edgecolor='black', linewidth=0.8, alpha=0.9)
            ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                        ecolor='#333333', capsize=4, capthick=1.2, elinewidth=1.2)
    
    ax.set_xlabel('Sample Group', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=font_size-1, fontweight='semibold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, fontsize=font_size-1)
    ax.set_title(title, fontsize=font_size+2, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def plot_scatter_plot(df, x_element, y_element, font_size, group_by=False):
    df = get_grouped_data(df, group_by)
    fig, ax = plt.subplots(figsize=(11, 9))
    
    groups = df['Sample'].str.extract(r'(CH|PH|CNH|PNH)')[0].fillna('Unknown')
    unique_groups = groups.unique()
    colors = get_safe_colors("colorblind", len(unique_groups))
    
    for group, color in zip(unique_groups, colors):
        mask = groups == group
        x_vals = np.asarray(df[mask][x_element].values).ravel()
        y_vals = np.asarray(df[mask][y_element].values).ravel()
        
        ax.scatter(x_vals, y_vals, label=group, s=120, alpha=0.85, 
                   color=color, edgecolors='black', linewidth=1.5, zorder=3)
        
        if len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), '--', color=color, alpha=0.7, linewidth=1.5, zorder=2)
            try:
                cov = np.polyfit(x_vals, y_vals, 1, cov=True)[1]
                ci = np.polyval(cov, x_vals)
                ax.fill_between(x_vals, p(x_vals)-ci*1.96, p(x_vals)+ci*1.96, alpha=0.1, color=color)
            except:
                pass
    
    ax.set_xlabel(f'{x_element} (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(f'{y_element} (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_title(f'Correlation: {x_element} vs {y_element}', fontsize=font_size+2, fontweight='bold', pad=15)
    ax.legend(frameon=True, fontsize=font_size-1, loc='best', fancybox=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.3, zorder=1)
    
    try:
        corr, p_val = stats.pearsonr(df[x_element], df[y_element]) if SCIPY_AVAILABLE else (df[x_element].corr(df[y_element]), 0)
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}', transform=ax.transAxes, 
                fontsize=font_size, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    except:
        pass
    plt.tight_layout()
    return fig

def plot_radar_chart(df, sample_names, elements, font_size, group_by=False):
    df = get_grouped_data(df, group_by)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(elements), endpoint=False).tolist()
    angles += angles[:1]
    colors = get_safe_colors("tab10", len(sample_names))
    
    for idx, sample in enumerate(sample_names):
        if sample in df['Sample'].values:
            values = np.asarray(df[df['Sample'] == sample][elements].values).ravel().tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=sample, color=colors[idx], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(elements, fontsize=font_size-1, fontweight='bold')
    max_val = df[elements].max().max() * 1.2
    ax.set_ylim(0, max_val)
    ax.set_yticklabels([])
    ax.set_title('Element Profile Radar Chart', fontsize=font_size+2, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=font_size-1, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_violin_distribution(df, elements, font_size, group_by=False):
    df = get_grouped_data(df, group_by)
    df_melted = df.melt(id_vars=['Sample'], value_vars=elements, 
                        var_name='Element', value_name='Concentration')
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.violinplot(data=df_melted, x='Element', y='Concentration', 
                   palette='pastel', ax=ax, cut=0, inner='quartile', linewidth=1.5)
    sns.swarmplot(data=df_melted, x='Element', y='Concentration', 
                  color='black', alpha=0.7, size=5, ax=ax, edgecolor='white', linewidth=1)
    ax.set_xlabel('Element', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_title('Distribution of Element Concentrations', fontsize=font_size+2, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()
    return fig

# ============================================
# PLOTLY INTERACTIVE VISUALIZATIONS
# ============================================

def apply_plotly_style(fig, font_family, font_size, publication_style):
    template = "plotly_white" if publication_style else "plotly_dark"
    fig.update_layout(template=template,
                      font=dict(family=font_family, size=font_size),
                      title_font=dict(size=font_size+2, family=font_family),
                      plot_bgcolor="white" if publication_style else "#111111",
                      paper_bgcolor="white" if publication_style else "#111111")
    return fig

def plot_3d_scatter(df, x_element, y_element, z_element, font_size, font_family, publication_style):
    fig = px.scatter_3d(df, x=x_element, y=y_element, z=z_element,
                        color='Sample', size_max=12, opacity=0.9,
                        title=f'3D Visualization: {x_element} vs {y_element} vs {z_element}')
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    return apply_plotly_style(fig, font_family, font_size, publication_style)

def plot_heatmap(df, elements, font_size, font_family, publication_style):
    corr_matrix = df[elements].corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Element Correlation Heatmap',
                    labels=dict(color='Pearson r'))
    fig.update_layout(font=dict(family=font_family, size=font_size))
    fig.update_xaxes(side='bottom')
    fig.update_traces(textfont=dict(size=font_size-1, color='black'))
    return fig

def plot_sunburst(df, elements, font_size, font_family, publication_style):
    avg_values = df[elements].mean()
    fig = go.Figure(go.Sunburst(
        labels=['Total'] + elements,
        parents=[''] + ['Total'] * len(elements),
        values=[sum(avg_values)] + avg_values.tolist(),
        branchvalues="total",
        marker=dict(colors=px.colors.qualitative.Dark2),
        textinfo="label+percent entry",
        textfont=dict(family=font_family, size=font_size-1)
    ))
    fig.update_layout(title='Element Composition Sunburst Chart',
                      width=850, height=850,
                      font=dict(family=font_family, size=font_size))
    return fig

def plot_parallel_coordinates(df, elements, font_size, font_family, publication_style):
    fig = px.parallel_coordinates(df, dimensions=elements,
                                  color='Co' if 'Co' in df.columns else elements[0],
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  title='Parallel Coordinates: Multi-element Comparison')
    fig.update_layout(font=dict(family=font_family, size=font_size))
    return apply_plotly_style(fig, font_family, font_size, publication_style)

def plot_bubble_chart(df, x_element, y_element, size_element, font_size, font_family, publication_style):
    fig = px.scatter(df, x=x_element, y=y_element, size=size_element,
                     color='Sample', hover_name='Sample', size_max=35,
                     title=f'Bubble Chart: {x_element} vs {y_element} (size: {size_element})',
                     opacity=0.85)
    fig.update_traces(marker=dict(line=dict(width=1.5, color='black')))
    return apply_plotly_style(fig, font_family, font_size, publication_style)

def plot_donut_comparison(df_ringer, df_lac, element, font_size, font_family, publication_style):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=["Ringer's Solution", "Lactic Acid + NaCl"])
    labels = df_ringer['Sample'].values
    fig.add_trace(go.Pie(labels=labels, values=df_ringer[element].values, hole=0.45, name="Ringer's",
                         marker=dict(colors=px.colors.qualitative.Set2), textfont=dict(size=font_size-1)), row=1, col=1)
    fig.add_trace(go.Pie(labels=labels, values=df_lac[element].values, hole=0.45, name="Lactic Acid",
                         marker=dict(colors=px.colors.qualitative.Set2), textfont=dict(size=font_size-1)), row=1, col=2)
    fig.update_layout(title_text=f'{element} Distribution Comparison',
                      font=dict(family=font_family, size=font_size),
                      title_font_size=font_size+2, width=1250, height=650)
    return fig

def plot_solution_comparison(df_ringer, df_lac, element, font_size, publication_style):
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(df_ringer['Sample']))
    width = 0.32
    vals_ringer = np.asarray(df_ringer[element].values).ravel()
    vals_lac = np.asarray(df_lac[element].values).ravel()
    
    ax.bar(x - width/2, vals_ringer, width, label="Ringer's Solution",
           color='#3A86FF', edgecolor='black', linewidth=1, alpha=0.9)
    ax.bar(x + width/2, vals_lac, width, label="Lactic Acid + NaCl",
           color='#FF006E', edgecolor='black', linewidth=1, alpha=0.9)
    
    if f'{element}_err' in df_ringer.columns:
        ax.errorbar(x - width/2, vals_ringer, yerr=df_ringer[f'{element}_err'].values,
                    fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    if f'{element}_err' in df_lac.columns:
        ax.errorbar(x + width/2, vals_lac, yerr=df_lac[f'{element}_err'].values,
                    fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    
    ax.set_xlabel('Sample', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(f'{element} Concentration (mg/L)', fontsize=font_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_ringer['Sample'], rotation=45, ha='right', fontsize=font_size-1, fontweight='semibold')
    ax.legend(frameon=True, fontsize=font_size, loc='upper left')
    ax.set_title(f'{element} Release: Ringer\'s vs Lactic Acid Solution', 
                 fontsize=font_size+2, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================
# MAIN APP INTERFACE
# ============================================

st.title("🔬 ICP-AES Results: Advanced Visualization Suite")
st.markdown("### Cobalt-Chromium Alloys in Ringer's & Lactic Acid Solutions | Publication-Ready Engine")

# SIDEBAR CONTROLS
st.sidebar.header("🎛️ Appearance & Controls")

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    font_family = st.selectbox("Font Family", ["sans-serif", "Arial", "Times New Roman", "Helvetica", "Georgia", "Verdana", "Courier New"])
with col_f2:
    font_size = st.slider("Font Size (pts)", 8, 22, 13)

font_style = st.sidebar.selectbox("Font Weight/Style", ["Normal", "Bold", "Italic", "Bold-Italic"])
color_palettes = [
    "colorblind", "tab10", "tab20", "Set1", "Set2", "Set3", "pastel", "muted", 
    "dark", "bright", "viridis", "plasma", "cividis", "flare", "crest", "mako",
    "Wong (High-Contrast)", "Publication Standard"
]
color_palette = st.sidebar.selectbox("Color Palette", color_palettes)
publication_style = st.sidebar.toggle("Publication Quality Mode", value=True, 
                                      help="Removes spines, adjusts grid, optimizes for journals")

# Data Controls
st.sidebar.markdown("---")
st.sidebar.header("📊 Data Configuration")
group_by_prefix = st.sidebar.toggle("Group by Sample Prefix (CH0, PH0, etc.)", value=True)
timepoint = st.sidebar.selectbox("Time Point", ["7 days", "1 month"])
solution_type = st.sidebar.multiselect("Solutions to Compare", 
                                        ["Ringer's Solution", "Lactic Acid + NaCl"],
                                        default=["Ringer's Solution"])

# Load Data
df_ringer_7d = create_ringer_7d()
df_ringer_1m = create_ringer_1m()
df_lac_7d = create_lac_7d()
df_lac_1m = create_lac_1m()

# Apply styling globally
apply_matplotlib_styling(font_family, font_style, font_size, publication_style)

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Bar & Grouped Plots", 
    "🔬 Advanced Analytics", 
    "🎨 Modern Visualizations",
    "📈 Interactive Plotly",
    "⚖️ Solution Comparison"
])

# ============================================
# TAB 1: BAR & GROUPED PLOTS
# ============================================
with tab1:
    st.header("Grouped Bar Charts & Comparisons")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        avail = set()
        if "Ringer's Solution" in solution_type:
            df_t = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
            avail |= set([c for c in df_t.columns if not c.endswith('_err') and c != 'Sample'])
        if "Lactic Acid + NaCl" in solution_type:
            df_t = df_lac_7d if timepoint == "7 days" else df_lac_1m
            avail |= set([c for c in df_t.columns if not c.endswith('_err') and c != 'Sample'])
        avail = sorted(list(avail))
        selected = st.multiselect("Select Elements", avail, default=avail[:4])
        
    with col_b:
        plot_type = st.radio("Plot Type", ["Grouped Bars", "Scatter Plot", "Radar Chart", "Violin Plot"], horizontal=True)
    
    if selected:
        if plot_type == "Grouped Bars":
            for sol in solution_type:
                df_c = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if sol == "Ringer's Solution" else df_lac_7d if timepoint == "7 days" else df_lac_1m
                fig = plot_grouped_bars_matplotlib(df_c, selected, f"{sol} - {timepoint}", "Concentration (mg/L)", font_size, color_palette, group_by_prefix)
                st.pyplot(fig)
                if st.button(f"📥 Download {sol}", key=f"dl_{sol}_bar"):
                    st.markdown(download_figure_matplotlib(fig, f"{sol}_{timepoint.replace(' ', '_')}_bars"), unsafe_allow_html=True)
        
        elif plot_type == "Scatter Plot" and len(selected) >= 2:
            cx, cy = st.columns(2)
            with cx: xe = st.selectbox("X-axis", selected, key="sx")
            with cy: ye = st.selectbox("Y-axis", selected, key="sy")
            for sol in solution_type:
                df_c = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if sol == "Ringer's Solution" else df_lac_7d if timepoint == "7 days" else df_lac_1m
                fig = plot_scatter_plot(df_c, xe, ye, font_size, group_by_prefix)
                st.pyplot(fig)
                if st.button(f"📥 Download Scatter", key=f"dl_{sol}_sc"):
                    st.markdown(download_figure_matplotlib(fig, f"{sol}_{timepoint.replace(' ', '_')}_scatter"), unsafe_allow_html=True)
        
        elif plot_type == "Radar Chart":
            df_r = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
            samps = st.multiselect("Samples", df_r['Sample'].tolist(), default=df_r['Sample'].tolist()[:4])
            if samps:
                fig = plot_radar_chart(df_r, samps, selected, font_size, group_by_prefix)
                st.pyplot(fig)
        
        elif plot_type == "Violin Plot":
            for sol in solution_type:
                df_c = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if sol == "Ringer's Solution" else df_lac_7d if timepoint == "7 days" else df_lac_1m
                fig = plot_violin_distribution(df_c, selected, font_size, group_by_prefix)
                st.pyplot(fig)

# ============================================
# TAB 2: ADVANCED ANALYTICS
# ============================================
with tab2:
    st.header("🔬 Statistical & Distribution Analysis")
    
    if len(solution_type) > 0:
        df_c = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if "Ringer's Solution" in solution_type else df_lac_7d if timepoint == "7 days" else df_lac_1m
        elements = [c for c in df_c.columns if not c.endswith('_err') and c != 'Sample']
        analysis = st.selectbox("Analysis Type", ["Correlation Matrix", "Box Plots", "Statistical Summary", "Normality Test"])
        
        if analysis == "Correlation Matrix":
            sel = st.multiselect("Elements", elements, default=elements[:5])
            if len(sel) >= 2:
                corr = df_c[sel].corr()
                fig, ax = plt.subplots(figsize=(11, 9))
                sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, square=True, 
                           linewidths=1.5, cbar_kws={"shrink": 0.85}, ax=ax, fmt='.2f',
                           annot_kws={"size": font_size-1})
                ax.set_title('Pearson Correlation Matrix', fontsize=font_size+2, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis == "Box Plots":
            df_m = df_c.melt(id_vars=['Sample'], value_vars=elements[:6], var_name='Element', value_name='Concentration')
            fig, ax = plt.subplots(figsize=(13, 7))
            sns.boxplot(data=df_m, x='Element', y='Concentration', palette='pastel', ax=ax, showfliers=True, notch=True)
            sns.stripplot(data=df_m, x='Element', y='Concentration', color='black', alpha=0.6, size=5, ax=ax)
            ax.set_title('Concentration Distribution with Outliers', fontsize=font_size+2, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif analysis == "Statistical Summary":
            st.dataframe(df_c[elements].describe().round(4))
            cv = df_c[elements].std() / df_c[elements].mean() * 100
            st.subheader("Coefficient of Variation (%)")
            st.dataframe(cv.round(2))
        
        elif analysis == "Normality Test":
            if SCIPY_AVAILABLE:
                for el in elements[:4]:
                    try:
                        stat, p = stats.shapiro(df_c[el].dropna())
                        st.write(f"**{el}**: Shapiro-Wilk p={p:.4f} {'(Normal)' if p > 0.05 else '(Non-normal)'}")
                    except:
                        st.write(f"**{el}**: Insufficient data for normality test.")
            else:
                st.warning("scipy required for normality tests")

# ============================================
# TAB 3: MODERN VISUALIZATIONS
# ============================================
with tab3:
    st.header("🎨 Modern Publication Visualizations")
    
    df_mod = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if "Ringer's Solution" in solution_type else df_lac_7d if timepoint == "7 days" else df_lac_1m
    elements_mod = [c for c in df_mod.columns if not c.endswith('_err') and c != 'Sample']
    
    viz_type = st.selectbox("Visualization Type", ["Stacked Bar", "Line Chart", "Element Stacking %", "Pairplot"])
    
    if viz_type == "Stacked Bar":
        sel = st.multiselect("Elements to Stack", elements_mod, default=elements_mod[:4])
        if sel:
            fig, ax = plt.subplots(figsize=(13, 7))
            df_plot = df_mod.set_index('Sample')[sel]
            # FIXED: Uses safe color list instead of invalid colormap string
            safe_colors = get_safe_colors(color_palette, len(df_plot.columns))
            df_plot.plot(kind='bar', stacked=True, ax=ax, color=safe_colors, edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Concentration (mg/L)', fontsize=font_size, fontweight='bold')
            ax.set_title(f'Stacked Composition: {timepoint}', fontsize=font_size+2, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size-1)
            ax.grid(axis='y', alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif viz_type == "Line Chart":
        sel = st.multiselect("Elements", elements_mod, default=['Co', 'Cr'])
        if sel:
            fig, ax = plt.subplots(figsize=(13, 6))
            safe_colors = get_safe_colors(color_palette, len(sel))
            for idx, el in enumerate(sel):
                ax.plot(df_mod['Sample'], df_mod[el], marker='o', label=el, 
                        color=safe_colors[idx], linewidth=2.5, markersize=8, markeredgecolor='black')
            ax.set_ylabel('Concentration (mg/L)', fontsize=font_size, fontweight='bold')
            ax.set_title(f'Trend Analysis: {timepoint}', fontsize=font_size+2, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=font_size-1, frameon=True)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif viz_type == "Element Stacking %":
        df_pct = df_mod[elements_mod].div(df_mod[elements_mod].sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(13, 7))
        safe_colors = get_safe_colors("Set2", len(elements_mod))
        df_pct.plot(kind='bar', stacked=True, ax=ax, color=safe_colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Relative Composition (%)', fontsize=font_size, fontweight='bold')
        ax.set_title(f'Normalized Element Distribution: {timepoint}', fontsize=font_size+2, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size-1)
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_type == "Pairplot":
        sel = st.multiselect("Elements (max 5)", elements_mod, default=elements_mod[:3])
        if len(sel) >= 2:
            with st.spinner("Generating pairplot..."):
                g = sns.pairplot(df_mod[sel], diag_kind='kde', corner=False, plot_kws={'alpha': 0.7, 'edgecolor': 'black'}, height=3.5)
                g.fig.suptitle(f'Pairwise Relationships: {timepoint}', y=1.02, fontsize=font_size+2, fontweight='bold')
                st.pyplot(g.fig)

# ============================================
# TAB 4: INTERACTIVE PLOTLY
# ============================================
with tab4:
    st.header("📈 Interactive Plotly Visualizations")
    
    if not PLOTLY_AVAILABLE:
        st.error("⚠️ Install Plotly: `pip install plotly`")
    else:
        df_pl = df_ringer_7d if timepoint == "7 days" else df_ringer_1m if "Ringer's Solution" in solution_type else df_lac_7d if timepoint == "7 days" else df_lac_1m
        elements_pl = [c for c in df_pl.columns if not c.endswith('_err') and c != 'Sample']
        
        plotly_type = st.selectbox("Plotly Chart Type", ["3D Scatter", "Heatmap", "Sunburst", "Parallel Coordinates", "Bubble Chart", "Donut Comparison"])
        
        if plotly_type == "3D Scatter" and len(elements_pl) >= 3:
            x3, y3, z3 = st.columns(3)
            with x3: xe = st.selectbox("X", elements_pl, key="px")
            with y3: ye = st.selectbox("Y", elements_pl, key="py")
            with z3: ze = st.selectbox("Z", elements_pl, key="pz")
            fig = plot_3d_scatter(df_pl, xe, ye, ze, font_size, font_family, publication_style)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Heatmap":
            sel = st.multiselect("Elements", elements_pl, default=elements_pl[:5])
            if len(sel) >= 2:
                fig = plot_heatmap(df_pl, sel, font_size, font_family, publication_style)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Sunburst":
            fig = plot_sunburst(df_pl, elements_pl, font_size, font_family, publication_style)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Parallel Coordinates":
            fig = plot_parallel_coordinates(df_pl, elements_pl, font_size, font_family, publication_style)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Bubble Chart":
            bx, by, bs = st.columns(3)
            with bx: bxe = st.selectbox("X", elements_pl, key="bx")
            with by: bye = st.selectbox("Y", elements_pl, key="by")
            with bs: bse = st.selectbox("Size", elements_pl, key="bs")
            fig = plot_bubble_chart(df_pl, bxe, bye, bse, font_size, font_family, publication_style)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plotly_type == "Donut Comparison" and "Ringer's Solution" in solution_type and "Lactic Acid + NaCl" in solution_type:
            el_donut = st.selectbox("Element", elements_pl)
            df_r = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
            df_l = df_lac_7d if timepoint == "7 days" else df_lac_1m
            fig = plot_donut_comparison(df_r, df_l, el_donut, font_size, font_family, publication_style)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 5: SOLUTION COMPARISON
# ============================================
with tab5:
    st.header("⚖️ Solution Comparison")
    
    if "Ringer's Solution" not in solution_type or "Lactic Acid + NaCl" not in solution_type:
        st.warning("⚠️ Select both solutions in the sidebar for comparison.")
    else:
        df_r = df_ringer_7d if timepoint == "7 days" else df_ringer_1m
        df_l = df_lac_7d if timepoint == "7 days" else df_lac_1m
        common = sorted(list(set(df_r.columns) & set(df_l.columns) - {'Sample'} - {c for c in df_r.columns if c.endswith('_err')}))
        
        comp_type = st.selectbox("Comparison Type", ["Side-by-Side Bars", "Fold Change", "Statistical Summary"])
        
        if comp_type == "Side-by-Side Bars":
            el_comp = st.selectbox("Element to Compare", common)
            fig = plot_solution_comparison(df_r, df_l, el_comp, font_size, publication_style)
            st.pyplot(fig)
            if st.button(f"📥 Download Comparison"):
                st.markdown(download_figure_matplotlib(fig, f"comp_{el_comp}"), unsafe_allow_html=True)
        
        elif comp_type == "Fold Change":
            st.info("Fold Change = (Lactic Acid Mean) / (Ringer's Mean)")
            fc_data = {el: df_l[el].mean() / df_r[el].mean() for el in common}
            fc_df = pd.DataFrame(list(fc_data.items()), columns=['Element', 'Fold Change'])
            
            fig, ax = plt.subplots(figsize=(11, 6))
            bars = ax.bar(fc_df['Element'], fc_df['Fold Change'], 
                          color=fc_df['Fold Change'].apply(lambda x: '#FF006E' if x > 1 else '#3A86FF'),
                          edgecolor='black', linewidth=1, alpha=0.9)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='No Change')
            ax.set_ylabel('Fold Change (LAC / Ringer)', fontsize=font_size, fontweight='bold')
            ax.set_title(f'Element Release Fold Change: {timepoint}', fontsize=font_size+2, fontweight='bold')
            ax.legend(fontsize=font_size-1, frameon=True)
            ax.grid(axis='y', alpha=0.25)
            for bar, val in zip(bars, fc_df['Fold Change']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{val:.2f}x', ha='center', va='bottom', fontsize=font_size-1, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif comp_type == "Statistical Summary":
            st.subheader("Ringer's Solution")
            st.dataframe(df_r[common].describe().round(4))
            st.subheader("Lactic Acid + NaCl")
            st.dataframe(df_l[common].describe().round(4))
            
            comp_table = pd.DataFrame({
                'Element': common,
                'Ringer Mean': df_r[common].mean().round(4),
                'LAC Mean': df_l[common].mean().round(4),
                'Difference': (df_l[common].mean() - df_r[common].mean()).round(4),
                'Fold Change': (df_l[common].mean() / df_r[common].mean()).round(3)
            })
            st.dataframe(comp_table)

st.markdown("---")
st.caption("📊 ICP-AES Visualization Suite v2.3 | Supports: streamlit, pandas, numpy, matplotlib, seaborn, plotly, kaleido, scipy")
