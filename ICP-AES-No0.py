import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Page configuration
st.set_page_config(page_title="ICP-AES Publication Graphs", layout="wide")
st.title("📊 ICP-AES Results: Static Immersion Test")
st.markdown("### Cobalt-Chromium Alloys in Ringer's & Lactic Acid Solutions")

# ============================================
# DATA ENTRY (Hardcoded from your DOCX file)
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
# PLOTTING FUNCTIONS
# ============================================

def plot_grouped_bars(df, elements, title, ylabel, color_palette='viridis'):
    """Create grouped bar plot with error bars"""
    samples = df['Sample'].tolist()
    x = np.arange(len(samples))
    width = 0.8 / len(elements)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette(color_palette, len(elements))
    
    for i, element in enumerate(elements):
        values = df[element].values
        errors = df[f'{element}_err'].values
        offset = (i - len(elements)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=element, 
                      color=colors[i], edgecolor='black', linewidth=0.5)
        ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                    ecolor='black', capsize=3, capthick=1)
    
    ax.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_time_comparison(df_7d, df_1m, element, title, ylabel):
    """Compare 7-day vs 1-month concentrations"""
    samples = df_7d['Sample'].tolist()
    x = np.arange(len(samples))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    vals_7d = df_7d[element].values
    err_7d = df_7d[f'{element}_err'].values
    vals_1m = df_1m[element].values
    err_1m = df_1m[f'{element}_err'].values
    
    bars1 = ax.bar(x - width/2, vals_7d, width, label='7 days', 
                   color='#4682B4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, vals_1m, width, label='1 month',
                   color='#B22222', edgecolor='black', linewidth=0.5)
    
    ax.errorbar(x - width/2, vals_7d, yerr=err_7d, fmt='none', 
                ecolor='black', capsize=3, capthick=1)
    ax.errorbar(x + width/2, vals_1m, yerr=err_1m, fmt='none',
                ecolor='black', capsize=3, capthick=1)
    
    ax.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=9)
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def download_figure(fig, filename):
    """Convert figure to downloadable PNG"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download {filename}.png</a>'
    return href

# ============================================
# MAIN APP INTERFACE
# ============================================

# Sidebar controls
st.sidebar.header("🎛️ Plot Controls")
solution = st.sidebar.selectbox("Select Solution", ["Ringer's Solution", "Lactic Acid + NaCl"])
time_comparison = st.sidebar.checkbox("Compare 7-day vs 1-month", value=True)

# Available elements per dataset
if solution == "Ringer's Solution":
    df_7d = create_ringer_7d()
    df_1m = create_ringer_1m()
    available_elements = ['Co', 'Cr', 'W', 'Mo']
    if not time_comparison:
        available_elements += ['Si', 'Al']  # Only in 1M for Ringer
    ylabel = "Concentration (mg/L)"
else:
    df_7d = create_lac_7d()
    df_1m = create_lac_1m()
    available_elements = ['Co', 'Cr', 'Si', 'W', 'Mo', 'Al', 'Fe']
    ylabel = "Concentration (mg/L)"

selected_elements = st.sidebar.multiselect("Select Elements to Plot", available_elements, default=available_elements[:3])
color_palette = st.sidebar.selectbox("Color Palette", ["viridis", "plasma", "Set2", "tab10", "Set1"])

st.sidebar.markdown("---")
st.sidebar.info("📌 **Publication Tips:**\n- Use 300 dpi PNG\n- Fonts: Arial/Helvetica\n- For grayscale: use 'Set2' or 'tab10'")

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Bar Plots", "⏱️ Time Comparison", "📈 Trends Over Time"])

with tab1:
    st.subheader(f"{solution} - Element Concentrations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 7 Days")
        if selected_elements:
            fig7d = plot_grouped_bars(df_7d, selected_elements, 
                                      f"{solution} - 7 Days", ylabel, color_palette)
            st.pyplot(fig7d)
            
            if st.button("Download 7D Plot", key="download_7d"):
                st.markdown(download_figure(fig7d, f"{solution.replace(' ', '_')}_7D"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 1 Month")
        if selected_elements:
            # Filter to elements available in 1M
            avail_1m = [e for e in selected_elements if e in df_1m.columns]
            if avail_1m:
                fig1m = plot_grouped_bars(df_1m, avail_1m,
                                          f"{solution} - 1 Month", ylabel, color_palette)
                st.pyplot(fig1m)
                
                if st.button("Download 1M Plot", key="download_1m"):
                    st.markdown(download_figure(fig1m, f"{solution.replace(' ', '_')}_1M"), unsafe_allow_html=True)

with tab2:
    st.subheader(f"{solution} - 7 Days vs 1 Month Comparison")
    
    if selected_elements:
        element_compare = st.selectbox("Select Element to Compare", selected_elements)
        
        if element_compare in df_1m.columns:
            fig_comp = plot_time_comparison(df_7d, df_1m, element_compare,
                                            f"{solution} - {element_compare} Release Over Time",
                                            ylabel)
            st.pyplot(fig_comp)
            
            if st.button("Download Comparison Plot"):
                st.markdown(download_figure(fig_comp, f"{solution.replace(' ', '_')}_{element_compare}_comparison"), 
                           unsafe_allow_html=True)
        else:
            st.warning(f"{element_compare} not available in 1-month data for this solution")

with tab3:
    st.subheader("Concentration Trends (Line Plot)")
    st.info("Line plots showing release kinetics across all samples")
    
    if selected_elements:
        # Prepare time-series data
        time_points = ['7 days', '1 month']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for element in selected_elements:
            if element in df_1m.columns:
                vals_7d = df_7d[element].mean()  # Average across samples
                vals_1m = df_1m[element].mean()
                err_7d = df_7d[f'{element}_err'].mean()
                err_1m = df_1m[f'{element}_err'].mean()
                
                ax.plot(time_points, [vals_7d, vals_1m], 'o-', label=element, 
                       linewidth=2, markersize=8)
                ax.errorbar(time_points, [vals_7d, vals_1m], 
                           yerr=[[err_7d], [err_1m]], fmt='none', capsize=5)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f"{solution} - Average Element Release Over Time", 
                    fontsize=14, fontweight='bold')
        ax.legend(frameon=False, fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        st.pyplot(fig)
        
        if st.button("Download Trends Plot"):
            st.markdown(download_figure(fig, f"{solution.replace(' ', '_')}_trends"), unsafe_allow_html=True)

# ============================================
# DATA TABLE VIEWER
# ============================================
with st.expander("📋 View Raw Data"):
    st.subheader("7-Day Data")
    st.dataframe(df_7d)
    st.subheader("1-Month Data")
    st.dataframe(df_1m)

st.markdown("---")
st.caption("**Publication-ready plots** | Created for ICP-AES analysis | Use download buttons to save high-resolution images (300 DPI)")
