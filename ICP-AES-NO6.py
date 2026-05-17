import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import warnings
from matplotlib.ticker import MaxNLocator, ScalarFormatter, AutoMinorLocator
from matplotlib import rcParams

# Suppress non-critical matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib')

# Streamlit page configuration
st.set_page_config(page_title="Advanced Ion Release Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 ICP-AES Ion Release Analysis - Publication Quality Dashboard")

# ── Data (μg·cm⁻², converted from ppm × 10 mL / 10.2 cm²) ──────────────────
samples = ['CH0', 'CH45', 'PH0', 'PH45', 'CNH0', 'CNH45', 'PNH0', 'PNH45']
sample_labels = ['CH0', 'CH45', 'PH0', 'PH45', 'CNH0', 'CNH45', 'PNH0', 'PNH45']

data = {
    'Co': {
        'Ringer 7D': ([0.097, 0.110, 0.086, 0.093, 0.077, 0.080, 0.068, 0.070], [0.001]*8),
        'Ringer 1M': ([0.142, 0.164, 0.172, 0.187, 0.109, 0.120, 0.130, 0.142], [0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]),
        'Standard 7D':    ([1.422, 1.735, 1.853, 1.951, 1.196, 1.314, 1.304, 1.520], [0.001]*8),
        'Standard 1M':    ([1.873, 1.951, 2.069, 2.294, 1.088, 1.304, 1.324, 1.520], [0.010, 0.010, 0.029, 0.029, 0.020, 0.010, 0.020, 0.010]),
    },
    'Cr': {
        'Ringer 7D': ([0.067, 0.086, 0.075, 0.084, 0.043, 0.054, 0.063, 0.070], [0.001]*8),
        'Ringer 1M': ([0.086, 0.096, 0.097, 0.099, 0.067, 0.086, 0.076, 0.083], [0.001]*8),
        'Standard 7D':    ([0.873, 0.093, 0.097, 0.108, 0.073, 0.078, 0.075, 0.083], [0.002]*8),
        'Standard 1M':    ([0.099, 0.109, 0.207, 0.227, 0.092, 0.096, 0.066, 0.070], [0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001]),
    },
    'W': {
        'Ringer 7D': ([0.043, 0.054, 0.050, 0.054, 0.022, 0.030, 0.028, 0.032], [0.001]*8),
        'Ringer 1M': ([0.065, 0.070, 0.076, 0.079, 0.054, 0.058, 0.060, 0.064], [0.001]*8),
        'Standard 7D':    ([0.068, 0.073, 0.084, 0.086, 0.043, 0.047, 0.050, 0.054], [0.001]*8),
        'Standard 1M':    ([0.075, 0.083, 0.083, 0.089, 0.054, 0.059, 0.056, 0.064], [0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001]),
    },
    'Mo': {
        'Ringer 7D': ([0.022, 0.030, 0.025, 0.034, 0.011, 0.020, 0.015, 0.025], [0.001]*8),
        'Ringer 1M': ([0.043, 0.049, 0.054, 0.058, 0.022, 0.028, 0.034, 0.039], [0.001]*8),
        'Standard 7D':    ([0.045, 0.048, 0.050, 0.053, 0.022, 0.028, 0.032, 0.040], [0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001]),
        'Standard 1M':    ([0.037, 0.044, 0.048, 0.054, 0.025, 0.029, 0.022, 0.030], [0.001]*8),
    },
    'Si': {
        'Ringer 7D': None,
        'Ringer 1M': ([0.012, 0.019, 0.020, 0.021, 0.009, 0.010, 0.010, 0.020], [0.001]*8),
        'Standard 7D':    ([0.033, 0.037, 0.043, 0.050, 0.022, 0.028, 0.028, 0.030], [0.001]*8),
        'Standard 1M':    ([0.054, 0.068, 0.064, 0.071, 0.048, 0.054, 0.041, 0.048], [0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]),
    },
    'Al': {
        'Ringer 7D': None,
        'Ringer 1M': ([0.009, 0.010, 0.010, 0.015, None, None, None, None], [0.001, 0.001, 0.001, 0.001, None, None, None, None]),
        'Standard 7D':    None,
        'Standard 1M':    ([0.022, 0.025, 0.028, 0.030, 0.012, 0.015, 0.022, 0.027], [0.001]*8),
    },
}

# ── Publication Style Presets (Origin Pro Inspired) ─────────────────────────
PUBLICATION_STYLES = {
    "Origin Pro Default": {'font_family': 'sans-serif', 'font_size_base': 12, 'line_width': 1.0, 'marker_size': 5, 'dpi': 300, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'in', 'tick_length': 4},
    "Nature":  {'font_family': 'sans-serif', 'font_size_base': 9,  'line_width': 0.75,'marker_size': 3, 'dpi': 600, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'in', 'tick_length': 3},
    "Science": {'font_family': 'serif',      'font_size_base': 10, 'line_width': 0.8, 'marker_size': 4, 'dpi': 600, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'in', 'tick_length': 4},
    "IEEE":    {'font_family': 'sans-serif', 'font_size_base': 10, 'line_width': 1.0, 'marker_size': 5, 'dpi': 300, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'out', 'tick_length': 4},
    "APA":     {'font_family': 'serif',      'font_size_base': 12, 'line_width': 0.8, 'marker_size': 4, 'dpi': 300, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'in', 'tick_length': 4},
    "ACS":     {'font_family': 'serif',      'font_size_base': 10, 'line_width': 0.9, 'marker_size': 4, 'dpi': 600, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'in', 'tick_length': 3.5},
    "RSC":     {'font_family': 'sans-serif', 'font_size_base': 9,  'line_width': 0.8, 'marker_size': 3, 'dpi': 600, 'facecolor': '#FFFFFF', 'edgecolor': '#000000', 'tick_direction': 'in', 'tick_length': 3},
    "Dark Mode":{'font_family': 'sans-serif', 'font_size_base': 11, 'line_width': 0.8, 'marker_size': 4, 'dpi': 300, 'facecolor': '#1E1E1E', 'edgecolor': '#FFFFFF', 'tick_direction': 'in', 'tick_length': 4}
}

# Extended Colorblind-Friendly & Publication Palettes
COLORBLIND_PALETTES = {
    "Origin Default": ['#0072BD', '#D95319', '#EDB120', '#77AC30', '#4DBEEE', '#A2142F'],
    "Colorblind-Friendly": ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442'],
    "High-Contrast": ['#000000', '#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2'],
    "Monochrome": ['#333333', '#666666', '#999999', '#BBBBBB', '#DDDDDD', '#EEEEEE'],
    "Vibrant": ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A', '#2A9D8F'],
    "Viridis": ['#440154', '#31688E', '#35B779', '#FDE725'],
    "Plasma": ['#0D0887', '#6A00A8', '#B12A90', '#F0703A', '#FCF44A'],
    "Inferno": ['#000004', '#420A68', '#9A2B6B', '#DD4F43', '#FCFFA4'],
    "Magma": ['#000004', '#3B0F70', '#8C2981', '#DD4F43', '#FCFFA4'],
    "Cividis": ['#00224E', '#3E598C', '#7F8FAE', '#C1C5D0', '#FDE725'],
    "Turbo": ['#300040', '#0040FF', '#00FFFF', '#80FF00', '#FFFF00', '#FF8000', '#FF0000'],
    "Tableau 10": ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AC'],
    "Pastel": ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00'],
    "Set2": ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']
}

# Matplotlib sequential colormaps for continuous data
MATPLOTLIB_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'crest', 'flare', 'rocket', 'mako', 'icefire', 'vlag']

CHART_TYPES = ["Grouped Bar", "Stacked Bar", "Line Plot", "Scatter Plot", "Box Plot", "3D Bar", "Error Band"]

# ── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎨 Advanced Chart Settings")
    
    pub_style = st.selectbox("📚 Publication Style Preset", list(PUBLICATION_STYLES.keys()))
    chart_type = st.selectbox("📊 Chart Type", CHART_TYPES)
    
    use_3d = st.checkbox("🧊 Enable 3D Rendering", value=False, disabled=(chart_type != "3D Bar"))
    elev = st.slider("Elevation Angle", 0, 90, 30, 5, disabled=(not use_3d))
    azim = st.slider("Azimuth Angle", 0, 360, 45, 10, disabled=(not use_3d))
    
    st.subheader("🔤 Typography")
    # Font family with Calibri and Times New Roman options
    font_family_options = ["sans-serif", "serif", "monospace", "Calibri", "Times New Roman", "Arial", "Helvetica", "DejaVu Sans", "DejaVu Serif"]
    font_family = st.selectbox("Font Family", font_family_options, index=0)
    # Font size slider extended to 36
    font_size_base = st.slider("Base Font Size", 8, 36, PUBLICATION_STYLES[pub_style]['font_size_base'], 1)
    title_font_size = st.slider("Title Font Size", 10, 40, font_size_base + 2, 1)
    label_font_size = st.slider("Axis Label Font Size", 8, 36, font_size_base + 1, 1)
    tick_font_size = st.slider("Tick Label Font Size", 6, 30, font_size_base, 1)
    title_bold = st.checkbox("Bold Titles", value=True)
    label_italic = st.checkbox("Italic Axis Labels", value=False)
    use_math_text = st.checkbox("Use Math Text for Axes", value=True)
    
    st.subheader("🎨 Color Scheme")
    color_scheme_type = st.radio("Color Source", ["Categorical Palette", "Matplotlib Colormap"], index=0)
    
    if color_scheme_type == "Categorical Palette":
        color_palette = st.selectbox("Color Palette", list(COLORBLIND_PALETTES.keys()))
        custom_colors = {}
        conditions = ['Ringer 7D', 'Ringer 1M', 'Standard 7D', 'Standard 1M']
        default_palette = COLORBLIND_PALETTES[color_palette]
        for i, cond in enumerate(conditions):
            custom_colors[cond] = st.color_picker(cond, default_palette[i % len(default_palette)])
    else:
        selected_cmap = st.selectbox("Matplotlib Colormap", MATPLOTLIB_CMAPS)
        cmap = plt.get_cmap(selected_cmap)
        conditions = ['Ringer 7D', 'Ringer 1M', 'Standard 7D', 'Standard 1M']
        custom_colors = {cond: mcolors.rgb2hex(cmap(i/(len(conditions)-1))) for i, cond in enumerate(conditions)}
        st.write("🎨 Generated colors from colormap:")
        for cond, color in custom_colors.items():
            st.markdown(f"<span style='color:{color}'>●</span> {cond}: `{color}`", unsafe_allow_html=True)
    
    st.subheader("🖼️ Background & Frame")
    fig_bg = st.color_picker("Figure Background", PUBLICATION_STYLES[pub_style]['facecolor'])
    axes_bg = st.color_picker("Plot Area Background", "#FFFFFF")
    frame_style = st.radio("Frame Style", ["Full Box", "Bottom-Left Only", "No Frame", "Origin Style (inward ticks)"], index=0)
    frame_width = st.slider("Frame Line Width", 0.5, 2.5, 1.0, 0.1)
    frame_color = st.color_picker("Frame Color", "#333333")
    
    st.subheader("📐 Grid Configuration")
    show_grid = st.checkbox("Show Grid", value=True)
    if show_grid:
        grid_axis = st.radio("Grid Axis", ["Y-axis", "X-axis", "Both"], index=0)
        grid_style = st.selectbox("Grid Line Style", ["--", "-", ":", "-."], index=0)
        grid_width = st.slider("Grid Width", 0.1, 2.0, 0.4, 0.1)
        grid_alpha = st.slider("Grid Opacity", 0.0, 1.0, 0.3, 0.1)
        grid_color = st.color_picker("Grid Color", "#DDDDDD")
        show_minor_grid = st.checkbox("Show Minor Grid Lines", value=True)
        minor_grid_alpha = st.slider("Minor Grid Opacity", 0.0, 1.0, 0.15, 0.05)
    else:
        grid_axis, grid_style, grid_width, grid_alpha, grid_color, show_minor_grid, minor_grid_alpha = "y", "--", 0.4, 0.3, "#DDDDDD", False, 0.15
    
    st.subheader("📜 Legend")
    show_legend = st.checkbox("Show Legend", value=True)
    if show_legend:
        legend_pos = st.selectbox("Position", ["upper right", "upper left", "lower left", "lower right", "center right", "center left", "best", "outside right"], index=0)
        legend_frame = st.checkbox("Legend Frame", value=False)
        legend_frame_alpha = st.slider("Legend Frame Opacity", 0.0, 1.0, 0.9, 0.1) if legend_frame else 1.0
        legend_cols = st.slider("Legend Columns", 1, 4, 2)
        legend_font_size = st.slider("Legend Font Size", 6, 24, max(8, font_size_base - 2), 1)
        legend_marker_scale = st.slider("Legend Marker Scale", 0.5, 3.0, 1.5, 0.1)
    else:
        legend_pos, legend_frame, legend_cols, legend_font_size, legend_marker_scale = "upper right", False, 2, 8, 1.5
    
    st.subheader("📈 Data Display")
    show_error_bars = st.checkbox("Show Error Bars", value=True)
    error_cap_size = st.slider("Error Bar Cap Size", 2, 12, 5, 1)
    error_bar_width = st.slider("Error Bar Line Width", 0.5, 2.5, 1.0, 0.1)
    show_data_labels = st.checkbox("Show Data Labels", value=False)
    label_precision = st.slider("Label Decimal Places", 1, 4, 3, 1)
    label_font_size_data = st.slider("Data Label Font Size", 6, 20, max(7, font_size_base - 3), 1)
    label_rotation = st.slider("Data Label Rotation", -90, 90, 90, 15)
    
    st.subheader("📏 Axis Configuration")
    y_log_scale = st.checkbox("Log Scale Y-Axis", value=False)
    x_log_scale = st.checkbox("Log Scale X-Axis", value=False)
    auto_ylim = st.checkbox("Auto Y-Limits", value=True)
    y_min = st.number_input("Y-Axis Min", value=0.0, step=0.01, format="%.3f", disabled=auto_ylim)
    y_max = st.number_input("Y-Axis Max", value=2.5, step=0.01, format="%.3f", disabled=auto_ylim)
    show_scientific_notation = st.checkbox("Scientific Notation for Large/Small Values", value=True)
    offset_notation = st.checkbox("Use Offset Notation", value=True)
    
    st.subheader("✨ Advanced Styling")
    bar_hatch = st.selectbox("Bar Hatch Pattern", ["none", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"], index=0)
    bar_alpha = st.slider("Bar Transparency", 0.3, 1.0, 0.95, 0.05)
    bar_edge_width = st.slider("Bar Edge Width", 0.0, 2.0, 0.6, 0.1)
    bar_edge_color = st.color_picker("Bar Edge Color", "#FFFFFF")
    marker_edge_width = st.slider("Marker Edge Width", 0.0, 2.0, 0.8, 0.1)
    marker_edge_color = st.color_picker("Marker Edge Color", "#000000")
    
    # Origin Pro style enhancements
    panel_labels = st.checkbox("Add Panel Labels (a), (b), (c)...", value=False)
    watermark_text = st.text_input("Watermark Text (optional)", "")
    watermark_alpha = st.slider("Watermark Opacity", 0.0, 1.0, 0.1, 0.05) if watermark_text else 0.0
    
    # Tick customization
    tick_direction = st.radio("Tick Direction", ["in", "out", "inout"], index=0)
    tick_length = st.slider("Major Tick Length", 2, 10, 4, 1)
    minor_tick_length = st.slider("Minor Tick Length", 1, 6, 2, 1)
    show_minor_ticks = st.checkbox("Show Minor Ticks", value=True)
    
    view_mode = st.radio("📋 View Mode", ["Individual Charts", "Combined Grid (2×3)"], index=0)
    
    st.subheader("💾 Export")
    export_dpi = st.slider("Export DPI", 150, 1200, PUBLICATION_STYLES[pub_style]['dpi'], 50)
    export_format = st.selectbox("Preferred Format Info", ["PNG (raster)", "PDF (vector)", "SVG (vector)", "EPS (vector)"], index=1)
    if st.button("📥 Export Tips"):
        st.info("💡 **Pro Tips:**\n\n• Right-click any chart > 'Save Image As' for PNG\n• Use browser Print (Ctrl+P) > 'Save as PDF' for vector quality\n• For publication: use 600+ DPI for raster, or PDF/SVG for vector\n• Origin Pro style: use serif fonts + inward ticks + subtle grid")

# ── Apply Publication Style (Origin Pro Inspired) ───────────────────────────
style = PUBLICATION_STYLES[pub_style]

# Font mapping for Calibri and Times New Roman
font_mapping = {
    "Calibri": ["Calibri", "sans-serif"],
    "Times New Roman": ["Times New Roman", "serif"],
    "Arial": ["Arial", "sans-serif"],
    "Helvetica": ["Helvetica", "sans-serif"],
    "DejaVu Sans": ["DejaVu Sans", "sans-serif"],
    "DejaVu Serif": ["DejaVu Serif", "serif"]
}

# Determine actual font family for matplotlib
if font_family in font_mapping:
    mpl_font_family = font_mapping[font_family][0]
    mpl_font_generic = font_mapping[font_family][1]
else:
    mpl_font_family = font_family if font_family in ["sans-serif", "serif", "monospace"] else "sans-serif"
    mpl_font_generic = font_family if font_family in ["sans-serif", "serif", "monospace"] else "sans-serif"

# Update matplotlib rcParams with Origin Pro inspired settings
plt.rcParams.update({
    'font.family': mpl_font_family if font_family in font_mapping else (font_family if font_family in ["sans-serif", "serif", "monospace"] else style['font_family']),
    'font.sans-serif': ['Calibri', 'Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans', 'sans-serif'],
    'font.serif': ['Times New Roman', 'STIX', 'DejaVu Serif', 'serif'],
    'font.monospace': ['Consolas', 'DejaVu Sans Mono', 'monospace'],
    'font.size': font_size_base,
    'axes.linewidth': frame_width,
    'axes.labelsize': label_font_size,
    'axes.titlesize': title_font_size,
    'axes.titleweight': 'bold' if title_bold else 'normal',
    'axes.labelweight': 'bold' if label_italic else 'normal',
    'xtick.labelsize': tick_font_size,
    'ytick.labelsize': tick_font_size,
    'xtick.major.width': frame_width,
    'ytick.major.width': frame_width,
    'xtick.major.size': tick_length,
    'ytick.major.size': tick_length,
    'xtick.minor.size': minor_tick_length if show_minor_ticks else 0,
    'ytick.minor.size': minor_tick_length if show_minor_ticks else 0,
    'xtick.minor.visible': show_minor_ticks,
    'ytick.minor.visible': show_minor_ticks,
    'xtick.direction': tick_direction,
    'ytick.direction': tick_direction,
    'axes.spines.top': frame_style in ["Full Box", "Origin Style (inward ticks)"],
    'axes.spines.right': frame_style in ["Full Box", "Origin Style (inward ticks)"],
    'axes.spines.left': frame_style != "No Frame",
    'axes.spines.bottom': frame_style != "No Frame",
    'figure.facecolor': fig_bg,
    'axes.facecolor': axes_bg,
    'figure.dpi': export_dpi,
    'savefig.dpi': export_dpi,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.prop_cycle': plt.cycler(color=list(COLORBLIND_PALETTES["Origin Default"])),
})

# ── Plot Generation Helper (Origin Pro Quality) ─────────────────────────────
def render_chart(ax, element, cond_data, chart_type, use_3d=False, is_combined=False, panel_label=None):
    x_positions = np.arange(len(samples))
    plot_data = {}
    for cond in conditions:
        vals_errs = cond_data[cond]
        if vals_errs is None:
            continue
        vals, errs = vals_errs
        plot_vals = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        plot_errs = np.array([e if e is not None else np.nan for e in errs], dtype=float)
        plot_data[cond] = (plot_vals, plot_errs)
    
    detected_conditions = list(plot_data.keys())
    if not detected_conditions:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=font_size_base)
        return
    
    n_cond = len(detected_conditions)
    bar_w = 0.18
    offsets = np.arange(n_cond) * bar_w - (n_cond - 1) * bar_w / 2
    
    # Panel label for multi-panel figures (Origin Pro style)
    if panel_label and is_combined:
        ax.text(-0.15, 1.05, f'{panel_label})', transform=ax.transAxes, 
               fontsize=title_font_size, fontweight='bold', va='bottom', ha='right')
    
    # ── 3D Bar Rendering ────────────────────────────────────────────────
    if chart_type == "3D Bar" and use_3d:
        from mpl_toolkits.mplot3d import Axes3D
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            xs = x_positions[valid_mask]
            ys = np.full_like(xs, i)
            zs = np.zeros_like(xs)
            dx = np.full_like(xs, 0.8)
            dy = np.full_like(xs, 0.8)
            dz = vals[valid_mask]
            ax.bar3d(xs, ys, zs, dx, dy, dz, color=custom_colors[cond], 
                     alpha=bar_alpha, edgecolor=bar_edge_color, linewidth=bar_edge_width)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sample_labels, fontsize=tick_font_size, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_cond))
        ax.set_yticklabels(detected_conditions, fontsize=tick_font_size)
        ax.set_zlabel('Ion release (μg·cm⁻²)', fontsize=label_font_size, fontstyle='italic' if label_italic else 'normal', labelpad=8)
        ax.set_title(f'{element} ion release', fontsize=title_font_size, fontweight='bold' if title_bold else 'normal', pad=15)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(axis='both', which='major', direction=tick_direction, length=tick_length, width=frame_width)
        return
    
    # ── 2D Chart Rendering ──────────────────────────────────────────────
    if chart_type == "Grouped Bar":
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            x_valid = x_positions[valid_mask] + offsets[i]
            bars = ax.bar(x_valid, vals[valid_mask], width=bar_w,
                   color=custom_colors[cond], label=cond, zorder=3,
                   edgecolor=bar_edge_color, linewidth=bar_edge_width, alpha=bar_alpha,
                   hatch=bar_hatch if bar_hatch != "none" else None)
            # Add marker-style edges to bars for Origin Pro look
            for bar in bars:
                bar.set_edgecolor(bar_edge_color)
                bar.set_linewidth(bar_edge_width)
            if show_error_bars:
                ax.errorbar(x_valid, vals[valid_mask], yerr=errs[valid_mask],
                           fmt='none', ecolor=frame_color, elinewidth=error_bar_width,
                           capsize=error_cap_size, capthick=error_bar_width, zorder=4)
            if show_data_labels:
                for x_pos, val in zip(x_valid, vals[valid_mask]):
                    if not np.isnan(val) and val > 0:
                        ax.text(x_pos, val + 0.02, f'{val:.{label_precision}f}',
                               ha='center', va='bottom', fontsize=label_font_size_data, rotation=label_rotation,
                               fontweight='normal', color=frame_color)
    
    elif chart_type == "Stacked Bar":
        bottom = np.zeros(len(samples))
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            plot_vals = np.where(valid_mask, vals, 0)
            bars = ax.bar(x_positions, plot_vals, bottom=bottom,
                   color=custom_colors[cond], label=cond, zorder=3,
                   edgecolor=bar_edge_color, linewidth=bar_edge_width, alpha=bar_alpha,
                   hatch=bar_hatch if bar_hatch != "none" else None)
            for bar in bars:
                bar.set_edgecolor(bar_edge_color)
                bar.set_linewidth(bar_edge_width)
            bottom += plot_vals
            if show_error_bars and np.any(valid_mask):
                ax.errorbar(x_positions[valid_mask], bottom[valid_mask] - errs[valid_mask]/2,
                           yerr=errs[valid_mask], fmt='none', ecolor=frame_color,
                           elinewidth=error_bar_width, capsize=error_cap_size, zorder=4)
    
    elif chart_type == "Line Plot":
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            line, = ax.plot(x_positions[valid_mask], vals[valid_mask],
                   marker=markers[i % len(markers)], markersize=style['marker_size'] + 1,
                   label=cond, color=custom_colors[cond], linewidth=style['line_width']*1.5,
                   markeredgecolor=marker_edge_color, markeredgewidth=marker_edge_width,
                   markerfacecolor=custom_colors[cond])
            if show_error_bars:
                ax.errorbar(x_positions[valid_mask], vals[valid_mask], yerr=errs[valid_mask],
                           fmt='none', ecolor=custom_colors[cond], elinewidth=error_bar_width,
                           capsize=error_cap_size, alpha=0.8, capthick=error_bar_width)
    
    elif chart_type == "Scatter Plot":
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            jitter = np.random.RandomState(42 + i).normal(0, 0.02, size=np.sum(valid_mask))
            ax.scatter(x_positions[valid_mask] + jitter, vals[valid_mask],
                      marker=markers[i % len(markers)], s=(style['marker_size']+2)*20,
                      label=cond, color=custom_colors[cond], alpha=bar_alpha, 
                      edgecolors=marker_edge_color, linewidths=marker_edge_width)
            if show_error_bars:
                ax.errorbar(x_positions[valid_mask], vals[valid_mask], yerr=errs[valid_mask],
                           fmt='none', ecolor=custom_colors[cond], elinewidth=error_bar_width,
                           capsize=error_cap_size, alpha=0.7, capthick=error_bar_width)
    
    elif chart_type == "Box Plot":
        box_data = []
        box_labels = []
        for cond in detected_conditions:
            vals, _ = plot_data[cond]
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                box_data.append(valid_vals)
                box_labels.append(cond)
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6, showfliers=False,
                           medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(linewidth=frame_width),
                           whiskerprops=dict(linewidth=frame_width),
                           capprops=dict(linewidth=frame_width, capsize=error_cap_size))
            for patch, cond in zip(bp['boxes'], box_labels):
                patch.set_facecolor(custom_colors[cond])
                patch.set_alpha(bar_alpha)
                patch.set_edgecolor(bar_edge_color)
                patch.set_linewidth(bar_edge_width)
            ax.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=tick_font_size)
    
    elif chart_type == "Error Band":
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            x_valid = x_positions[valid_mask]
            ax.plot(x_valid, vals[valid_mask], label=cond, color=custom_colors[cond], 
                   linewidth=style['line_width']*1.5, marker='o', markersize=style['marker_size'],
                   markeredgecolor=marker_edge_color, markeredgewidth=marker_edge_width)
            ax.fill_between(x_valid, np.maximum(0, vals[valid_mask] - errs[valid_mask]),
                           vals[valid_mask] + errs[valid_mask], color=custom_colors[cond], alpha=0.2, label=None)

    # ── Axis Configuration (Origin Pro Style) ────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sample_labels, fontsize=tick_font_size, rotation=45, ha='right')
    ax.set_ylabel('Ion release (μg·cm⁻²)', fontsize=label_font_size, fontstyle='italic' if label_italic else 'normal', labelpad=8)
    ax.set_xlabel('Sample ID', fontsize=label_font_size, fontstyle='italic' if label_italic else 'normal', labelpad=8)
    ax.set_title(f'{element} ion release', fontsize=title_font_size, fontweight='bold' if title_bold else 'normal', pad=15, loc='left')
    
    if not auto_ylim:
        ax.set_ylim(y_min, y_max)
    if y_log_scale and chart_type not in ["Box Plot"]:
        ax.set_yscale('log')
    if x_log_scale:
        ax.set_xscale('log')
    
    # Scientific notation and offset formatting
    if show_scientific_notation or offset_notation:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text, useOffset=offset_notation))
        ax.yaxis.major.formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text, useOffset=offset_notation))
    
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
    if show_minor_ticks:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # ── Grid Configuration (Origin Pro Style) ────────────────────────────
    if show_grid and frame_style != "No Frame":
        axis_map = {"Y-axis": "y", "X-axis": "x", "Both": "both"}
        ax.grid(True, axis=axis_map[grid_axis], linestyle=grid_style, linewidth=grid_width, alpha=grid_alpha, color=grid_color, zorder=0)
        if show_minor_grid and show_minor_ticks:
            ax.grid(which='minor', axis=axis_map[grid_axis], linestyle=':', linewidth=grid_width*0.5, alpha=minor_grid_alpha, color=grid_color, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    
    # ── Frame Styling (Origin Pro: inward ticks, clean spines) ───────────
    if frame_style != "No Frame":
        for spine_name, spine in ax.spines.items():
            if frame_style == "Full Box" or spine_name in ['left', 'bottom']:
                spine.set_visible(True)
                spine.set_linewidth(frame_width)
                spine.set_color(frame_color)
            else:
                spine.set_visible(False)
    
    # ── Legend Configuration (Origin Pro Style) ──────────────────────────
    if show_legend and detected_conditions:
        handles = [mpatches.Patch(color=custom_colors[c], label=c, edgecolor=bar_edge_color, linewidth=0.5) for c in detected_conditions]
        # Handle outside legend positioning
        if legend_pos == "outside right":
            ax.legend(handles=handles, frameon=legend_frame, fontsize=legend_font_size,
                     loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, 
                     handlelength=1.8, handleheight=legend_marker_scale, framealpha=legend_frame_alpha)
        else:
            ax.legend(handles=handles, frameon=legend_frame, fontsize=legend_font_size,
                     loc=legend_pos, ncol=legend_cols, handlelength=1.8, 
                     handleheight=legend_marker_scale, framealpha=legend_frame_alpha)
    
    # ── Watermark (Origin Pro style subtle branding) ─────────────────────
    if watermark_text and watermark_alpha > 0:
        ax.text(0.5, 0.5, watermark_text, transform=ax.transAxes,
               fontsize=font_size_base*3, color=frame_color, alpha=watermark_alpha,
               ha='center', va='center', rotation=30, fontweight='bold')
    
    # ── Annotation for Partial Detection ─────────────────────────────────
    if element in ('Al', 'Si'):
        ax.text(0.99, 0.02, '* n.d. = not detected', transform=ax.transAxes,
               fontsize=max(7, tick_font_size-2), ha='right', va='bottom', color='gray',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=axes_bg, alpha=0.7, edgecolor=frame_color, linewidth=0.5))

# ── Streamlit Rendering ─────────────────────────────────────────────────────
elements_to_plot = list(data.keys())
conditions = ['Ringer 7D', 'Ringer 1M', 'Standard 7D', 'Standard 1M']

if view_mode == "Individual Charts":
    for idx, element in enumerate(elements_to_plot):
        with st.expander(f"📊 {element} Ion Release", expanded=True):
            if chart_type == "3D Bar" and use_3d:
                fig = plt.figure(figsize=(12, 8), facecolor=fig_bg)
                ax = fig.add_subplot(111, projection='3d', facecolor=axes_bg)
            else:
                fig, ax = plt.subplots(figsize=(12, 7), facecolor=fig_bg)
                ax.set_facecolor(axes_bg)
            
            render_chart(ax, element, data[element], chart_type, use_3d, is_combined=False, panel_label=None)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True, bbox_inches='tight')
            plt.close(fig)
else:
    st.subheader("📋 Combined Element Overview")
    if chart_type == "3D Bar" and use_3d:
        st.warning("⚠️ 3D charts work best in individual view. Rendering 2D fallback for grid view.")
        current_use_3d = False
    else:
        current_use_3d = use_3d
    
    fig = plt.figure(figsize=(20, 14), facecolor=fig_bg)
    
    for idx, element in enumerate(elements_to_plot):
        if chart_type == "3D Bar" and current_use_3d:
            ax = fig.add_subplot(2, 3, idx+1, projection='3d', facecolor=axes_bg)
        else:
            ax = fig.add_subplot(2, 3, idx+1, facecolor=axes_bg)
        ax.set_facecolor(axes_bg)
        # Add panel label (a), (b), (c)... for Origin Pro multi-panel style
        panel_label = chr(97 + idx) if panel_labels else None
        render_chart(ax, element, data[element], chart_type, current_use_3d, is_combined=True, panel_label=panel_label)
    
    # Hide unused subplots
    for idx in range(len(elements_to_plot), 6):
        ax = fig.add_subplot(2, 3, idx+1)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for suptitle if needed
    st.pyplot(fig, use_container_width=True, bbox_inches='tight')
    plt.close(fig)

# ── Footer & Export Info ────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Elements", len(elements_to_plot))
with col2:
    st.metric("Samples", len(samples))
with col3:
    st.metric("Conditions", len(conditions))
with col4:
    st.metric("Font Size", f"{font_size_base}pt")

st.caption(f"🎨 Style: {pub_style} | 🔤 Font: {font_family} ({font_size_base}pt) | 📊 Type: {chart_type} | 🧊 3D: {'Yes' if use_3d else 'No'} | 📐 DPI: {export_dpi} | 🖼️ Format: {export_format}")

# Quick style reference
with st.expander("📖 Origin Pro Style Guide"):
    st.markdown("""
    **✅ Publication-Ready Settings Applied:**
    - 🔹 **Fonts**: Calibri/Times New Roman support, sizes up to 36pt
    - 🔹 **Colors**: 13+ palettes including Origin Default, Colorblind-Friendly, Viridis, Tableau
    - 🔹 **Ticks**: Inward direction, minor ticks, customizable length
    - 🔹 **Grid**: Subtle major/minor grid with opacity control
    - 🔹 **Frames**: Origin-style spine configuration
    - 🔹 **Legends**: Outside positioning, marker scaling, frame opacity
    - 🔹 **Export**: Up to 1200 DPI, vector format support (PDF/SVG/EPS)
    - 🔹 **Panels**: Automatic (a), (b), (c) labeling for multi-figure layouts
    - 🔹 **Annotations**: Scientific notation, offset formatting, watermark support
    
    **💡 Pro Tip**: For Nature/Science submissions, use: *Serif font + 9-10pt base + 600 DPI + inward ticks + subtle grid*
    """)

st.success("✅ Charts rendered with Origin Pro-inspired publication quality. Right-click to save or use Print > Save as PDF for vector output.")
