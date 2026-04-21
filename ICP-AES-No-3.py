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
# SAMPLE CATEGORIZATION FUNCTIONS
# ============================================

def extract_category_from_name(sample_name, categorization_rule):
    """
    Extract category from sample name based on user-defined rules.
    
    Categorization rules:
    - 'prefix': Extract first part before '-' (e.g., CH0, PH45)
    - 'suffix': Extract last part after '-' (e.g., Ringer-7D, LAC-1M)
    - 'material': Extract material code (CH, PH, CNH, PNH)
    - 'condition': Extract condition (0, 45)
    - 'custom': User-defined position
    """
    if categorization_rule == 'prefix':
        return sample_name.split('-')[0] if '-' in sample_name else sample_name
    elif categorization_rule == 'suffix':
        return sample_name.split('-')[-1] if '-' in sample_name else sample_name
    elif categorization_rule == 'material':
        # Extract material prefix (CH, PH, CNH, PNH)
        parts = sample_name.split('-')
        if parts:
            material = parts[0][:2] if parts[0].startswith(('CH', 'PH')) else parts[0][:3] if parts[0].startswith(('CNH', 'PNH')) else parts[0]
            return material
        return sample_name
    elif categorization_rule == 'condition':
        # Extract condition number (0, 45)
        parts = sample_name.split('-')
        if parts:
            # Look for numbers in the first part
            import re
            numbers = re.findall(r'\d+', parts[0])
            return numbers[0] if numbers else 'unknown'
        return 'unknown'
    elif categorization_rule == 'custom':
        # Custom extraction - user provides position
        return sample_name  # Will be handled by custom position
    else:
        return sample_name

def apply_custom_categorization(df, position, delimiter='-'):
    """Apply custom categorization based on position in split string."""
    df_cat = df.copy()
    categories = []
    for sample in df_cat['Sample']:
        parts = sample.split(delimiter)
        if len(parts) > position:
            categories.append(parts[position])
        else:
            categories.append(sample)
    df_cat['Category'] = categories
    return df_cat

def get_available_categories(df, categorization_rule, custom_position=0, custom_delimiter='-'):
    """Get available categories based on the selected rule."""
    categories = set()
    for sample in df['Sample']:
        if categorization_rule == 'custom':
            parts = sample.split(custom_delimiter)
            if len(parts) > custom_position:
                categories.add(parts[custom_position])
            else:
                categories.add(sample)
        else:
            categories.add(extract_category_from_name(sample, categorization_rule))
    return sorted(list(categories))

def categorize_samples(df, categorization_rule, selected_categories=None, 
                       custom_position=0, custom_delimiter='-'):
    """
    Categorize samples and optionally filter by selected categories.
    """
    df_cat = df.copy()
    
    if categorization_rule == 'custom':
        df_cat = apply_custom_categorization(df_cat, custom_position, custom_delimiter)
    else:
        df_cat['Category'] = df_cat['Sample'].apply(
            lambda x: extract_category_from_name(x, categorization_rule)
        )
    
    # Filter by selected categories if provided
    if selected_categories and len(selected_categories) > 0:
        df_cat = df_cat[df_cat['Category'].isin(selected_categories)]
    
    return df_cat

def get_grouped_data_by_category(df, group_by_prefix=False, categorization_rule=None, 
                                 selected_categories=None, custom_position=0, custom_delimiter='-'):
    """
    Return data with grouping by either prefix or user-defined categories.
    Enhanced to support sample name categorization.
    """
    if categorization_rule and categorization_rule != 'none':
        # Apply user-defined categorization
        df_cat = categorize_samples(df, categorization_rule, selected_categories,
                                   custom_position, custom_delimiter)
        group_col = 'Category'
    elif group_by_prefix:
        # Use original prefix-based grouping
        df_cat = df.copy()
        df_cat['Category'] = df_cat['Sample'].apply(lambda x: x.split('-')[0])
        group_col = 'Category'
    else:
        # No grouping, return as is
        return df.copy()
    
    # Ensure categorical ordering
    if selected_categories:
        df_cat['Category'] = pd.Categorical(df_cat['Category'], categories=selected_categories, ordered=True)
    else:
        unique_cats = sorted(df_cat['Category'].unique())
        df_cat['Category'] = pd.Categorical(df_cat['Category'], categories=unique_cats, ordered=True)
    
    df_cat = df_cat.sort_values('Category')
    
    # Get value columns (exclude metadata columns)
    val_cols = [c for c in df_cat.columns if not c.endswith('_err') and c not in ['Sample', 'Category']]
    
    # Group by category
    result_data = {'Sample': []}  # 'Sample' will hold category names for compatibility
    for c in val_cols:
        result_data[c] = []
        result_data[f'{c}_err'] = []
    
    for category_name, group_df in df_cat.groupby('Category'):
        result_data['Sample'].append(category_name)
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

def get_grouped_data(df, group_by_prefix=False, categorization_rule=None, 
                     selected_categories=None, custom_position=0, custom_delimiter='-'):
    """Return data with optional grouping. Enhanced with categorization."""
    if categorization_rule and categorization_rule != 'none':
        return get_grouped_data_by_category(df, group_by_prefix, categorization_rule,
                                           selected_categories, custom_position, custom_delimiter)
    elif group_by_prefix:
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
    else:
        return df.copy()

def download_figure_matplotlib(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
    return href

def download_figure_plotly(fig, filename):
    if KALEIDO_AVAILABLE:
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Download {filename}.png</a>'
        return href
    return "⚠️ `kaleido` not installed for Plotly export"

# ============================================
# MATPLOTLIB VISUALIZATIONS
# ============================================

def plot_grouped_bars_matplotlib(df, elements, title, ylabel, font_size, color_palette, group_by=False, 
                                 categorization_rule=None, selected_categories=None, custom_position=0, custom_delimiter='-'):
    df = get_grouped_data(df, group_by, categorization_rule, selected_categories, custom_position, custom_delimiter)
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

def plot_scatter_plot(df, x_element, y_element, font_size, group_by=False,
                      categorization_rule=None, selected_categories=None, custom_position=0, custom_delimiter='-'):
    df = get_grouped_data(df, group_by, categorization_rule, selected_categories, custom_position, custom_delimiter)
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

def plot_radar_chart(df, sample_names, elements, font_size, group_by=False,
                     categorization_rule=None, selected_categories=None, custom_position=0, custom_delimiter='-'):
    df = get_grouped_data(df, group_by, categorization_rule, selected_categories, custom_position, custom_delimiter)
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

def plot_violin_distribution(df, elements, font_size, group_by=False,
                             categorization_rule=None, selected_categories=None, custom_position=0, custom_delimiter='-'):
    df = get_grouped_data(df, group_by, categorization_rule, selected_categories, custom_position, custom_delimiter)
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
                                  color='Co' if
