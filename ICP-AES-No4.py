import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

# Suppress non-critical matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib')

# Streamlit page configuration
st.set_page_config(page_title="Advanced Ion Release Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 ICP-AES Ion Release Analysis - Publication Quality Dashboard")

# ── Data (μg·cm⁻², converted from ppm × 10 mL / 10.2 cm²) ──────────────────
samples = ['CH0', 'CH45', 'PH0', 'PH45', 'CNH0', 'CNH45', 'PNH0', 'PNH45']
sample_labels = ['CH₀', 'CH₄₅', 'PH₀', 'PH₄₅', 'CNH₀', 'CNH₄₅', 'PNH₀', 'PNH₄₅']

data = {
    'Co': {
        'Ringer 7D': ([0.097, 0.110, 0.086, 0.093, 0.077, 0.080, 0.068, 0.070],
                      [0.001]*8),
        'Ringer 1M': ([0.142, 0.164, 0.172, 0.187, 0.109, 0.120, 0.130, 0.142],
                      [0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]),
        'LAC 7D':    ([1.422, 1.735, 1.853, 1.951, 1.196, 1.314, 1.304, 1.520],
                      [0.001]*8),
        'LAC 1M':    ([1.873, 1.951, 2.069, 2.294, 1.088, 1.304, 1.324, 1.520],
                      [0.010, 0.010, 0.029, 0.029, 0.020, 0.010, 0.020, 0.010]),
    },
    'Cr': {
        'Ringer 7D': ([0.067, 0.086, 0.075, 0.084, 0.043, 0.054, 0.063, 0.070],
                      [0.001]*8),
        'Ringer 1M': ([0.086, 0.096, 0.097, 0.099, 0.067, 0.086, 0.076, 0.083],
                      [0.001]*8),
        'LAC 7D':    ([0.873, 0.093, 0.097, 0.108, 0.073, 0.078, 0.075, 0.083],
                      [0.002]*8),
        'LAC 1M':    ([0.099, 0.109, 0.207, 0.227, 0.092, 0.096, 0.066, 0.070],
                      [0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001]),
    },
    'W': {
        'Ringer 7D': ([0.043, 0.054, 0.050, 0.054, 0.022, 0.030, 0.028, 0.032],
                      [0.001]*8),
        'Ringer 1M': ([0.065, 0.070, 0.076, 0.079, 0.054, 0.058, 0.060, 0.064],
                      [0.001]*8),
        'LAC 7D':    ([0.068, 0.073, 0.084, 0.086, 0.043, 0.047, 0.050, 0.054],
                      [0.001]*8),
        'LAC 1M':    ([0.075, 0.083, 0.083, 0.089, 0.054, 0.059, 0.056, 0.064],
                      [0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001]),
    },
    'Mo': {
        'Ringer 7D': ([0.022, 0.030, 0.025, 0.034, 0.011, 0.020, 0.015, 0.025],
                      [0.001]*8),
        'Ringer 1M': ([0.043, 0.049, 0.054, 0.058, 0.022, 0.028, 0.034, 0.039],
                      [0.001]*8),
        'LAC 7D':    ([0.045, 0.048, 0.050, 0.053, 0.022, 0.028, 0.032, 0.040],
                      [0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001]),
        'LAC 1M':    ([0.037, 0.044, 0.048, 0.054, 0.025, 0.029, 0.022, 0.030],
                      [0.001]*8),
    },
    'Si': {
        'Ringer 7D': None,
        'Ringer 1M': ([0.012, 0.019, 0.020, 0.021, 0.009, 0.010, 0.010, 0.020],
                      [0.001]*8),
        'LAC 7D':    ([0.033, 0.037, 0.043, 0.050, 0.022, 0.028, 0.028, 0.030],
                      [0.001]*8),
        'LAC 1M':    ([0.054, 0.068, 0.064, 0.071, 0.048, 0.054, 0.041, 0.048],
                      [0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]),
    },
    'Al': {
        'Ringer 7D': None,
        'Ringer 1M': ([0.009, 0.010, 0.010, 0.015, None, None, None, None],
                      [0.001, 0.001, 0.001, 0.001, None, None, None, None]),
        'LAC 7D':    None,
        'LAC 1M':    ([0.022, 0.025, 0.028, 0.030, 0.012, 0.015, 0.022, 0.027],
                      [0.001]*8),
    },
}

# ── Publication Style Presets ───────────────────────────────────────────────
PUBLICATION_STYLES = {
    "Default": {
        'font_family': 'sans-serif',
        'font_size_base': 11,
        'line_width': 0.8,
        'marker_size': 4,
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    "Nature": {
        'font_family': 'sans-serif',
        'font_size_base': 9,
        'line_width': 0.75,
        'marker_size': 3,
        'dpi': 600,
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    "Science": {
        'font_family': 'serif',
        'font_size_base': 10,
        'line_width': 0.8,
        'marker_size': 4,
        'dpi': 600,
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    "IEEE": {
        'font_family': 'sans-serif',
        'font_size_base': 10,
        'line_width': 1.0,
        'marker_size': 5,
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    "APA": {
        'font_family': 'serif',
        'font_size_base': 12,
        'line_width': 0.8,
        'marker_size': 4,
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    "Dark Mode": {
        'font_family': 'sans-serif',
        'font_size_base': 11,
        'line_width': 0.8,
        'marker_size': 4,
        'dpi': 300,
        'facecolor': '#1e1e1e',
        'edgecolor': '#ffffff'
    }
}

COLORBLIND_PALETTES = {
    "Standard": ['#85B7EB', '#185FA5', '#F0997B', '#D85A30'],
    "Colorblind-Friendly": ['#0072B2', '#009E73', '#D55E00', '#CC79A7'],
    "High-Contrast": ['#000000', '#E69F00', '#56B4E9', '#009E73'],
    "Monochrome": ['#333333', '#666666', '#999999', '#CCCCCC'],
    "Vibrant": ['#E63946', '#F4A261', '#2A9D8F', '#264653']
}

CHART_TYPES = ["Grouped Bar", "Stacked Bar", "Line Plot", "Scatter Plot", "Box Plot", "3D Bar", "Error Band"]

# ── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎨 Advanced Chart Settings")
    
    # Publication preset
    pub_style = st.selectbox("📚 Publication Style Preset", list(PUBLICATION_STYLES.keys()))
    
    # Chart type selector
    chart_type = st.selectbox("📊 Chart Type", CHART_TYPES)
    
    # 3D options
    use_3d = st.checkbox("🧊 Enable 3D Rendering", value=False, disabled=chart_type not in ["3D Bar"])
    if use_3d and chart_type == "3D Bar":
        elev = st.slider("Elevation Angle", 0, 90, 30, 5)
        azim = st.slider("Azimuth Angle", 0, 360, 45, 10)
    
    # Font settings
    st.subheader("🔤 Typography")
    font_family = st.radio("Font Family", ["sans-serif", "serif", "monospace"], index=0)
    font_size_base = st.slider("Base Font Size", 8, 16, PUBLICATION_STYLES[pub_style]['font_size_base'], 1)
    title_bold = st.checkbox("Bold Titles", value=True)
    label_italic = st.checkbox("Italic Axis Labels", value=False)
    
    # Color settings
    st.subheader("🎨 Color Scheme")
    color_palette = st.selectbox("Color Palette", list(COLORBLIND_PALETTES.keys()))
    custom_colors = {}
    conditions = ['Ringer 7D', 'Ringer 1M', 'LAC 7D', 'LAC 1M']
    for i, cond in enumerate(conditions):
        custom_colors[cond] = st.color_picker(cond, COLORBLIND_PALETTES[color_palette][i % len(COLORBLIND_PALETTES[color_palette])])
    
    # Background & Frame
    st.subheader("🖼️ Background & Frame")
    fig_bg = st.color_picker("Figure Background", PUBLICATION_STYLES[pub_style]['facecolor'])
    axes_bg = st.color_picker("Plot Area Background", "#FFFFFF")
    frame_style = st.radio("Frame Style", ["Full Box", "Bottom-Left Only", "No Frame"], index=0)
    frame_width = st.slider("Frame Line Width", 0.5, 2.5, 0.8, 0.1)
    frame_color = st.color_picker("Frame Color", "#333333")
    
    # Grid settings
    st.subheader("📐 Grid Configuration")
    show_grid = st.checkbox("Show Grid", value=True)
    if show_grid:
        grid_axis = st.radio("Grid Axis", ["Y-axis", "X-axis", "Both"], index=0)
        grid_style = st.selectbox("Grid Line Style", ["--", "-", ":", "-."], index=0)
        grid_width = st.slider("Grid Width", 0.1, 2.0, 0.5, 0.1)
        grid_alpha = st.slider("Grid Opacity", 0.0, 1.0, 0.4, 0.1)
        grid_color = st.color_picker("Grid Color", "#DDDDDD")
        show_minor_grid = st.checkbox("Show Minor Grid Lines", value=False)
    
    # Legend settings
    st.subheader("📜 Legend")
    show_legend = st.checkbox("Show Legend", value=True)
    if show_legend:
        legend_pos = st.selectbox("Position", ["upper right", "upper left", "lower left", "lower right", "center right", "best"], index=0)
        legend_frame = st.checkbox("Legend Frame", value=False)
        legend_cols = st.slider("Legend Columns", 1, 4, 2)
    
    # Data display options
    st.subheader("📈 Data Display")
    show_error_bars = st.checkbox("Show Error Bars", value=True)
    error_cap_size = st.slider("Error Bar Cap Size", 2, 8, 4, 1)
    show_data_labels = st.checkbox("Show Data Labels", value=False)
    label_precision = st.slider("Label Decimal Places", 1, 4, 3, 1)
    
    # Axis settings
    st.subheader("📏 Axis Configuration")
    y_log_scale = st.checkbox("Log Scale Y-Axis", value=False)
    y_min = st.number_input("Y-Axis Min", value=0.0, step=0.01, format="%.3f")
    y_max = st.number_input("Y-Axis Max", value=2.5, step=0.01, format="%.3f")
    auto_ylim = st.checkbox("Auto Y-Limits", value=True)
    
    # Advanced styling
    st.subheader("✨ Advanced Styling")
    bar_hatch = st.selectbox("Bar Hatch Pattern", ["none", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"], index=0)
    bar_alpha = st.slider("Bar Transparency", 0.3, 1.0, 1.0, 0.1)
    bar_edge_width = st.slider("Bar Edge Width", 0.0, 2.0, 0.4, 0.1)
    
    # View mode
    view_mode = st.radio("📋 View Mode", ["Individual Charts", "Combined Grid (2×3)"], index=0)
    
    # Export options
    st.subheader("💾 Export")
    export_format = st.selectbox("Export Format", ["PNG", "SVG", "PDF"], index=0)
    export_dpi = st.slider("Export DPI", 150, 600, PUBLICATION_STYLES[pub_style]['dpi'], 50)
    
    if st.button("📥 Download Current Chart"):
        st.info("Use the matplotlib toolbar in the rendered chart to save, or implement backend export logic.")

# ── Apply Publication Style ─────────────────────────────────────────────────
style = PUBLICATION_STYLES[pub_style]
rcParams.update({
    'font.family': font_family if pub_style == "Default" else style['font_family'],
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
    'font.serif': ['Times New Roman', 'STIX', 'DejaVu Serif'],
    'font.size': font_size_base,
    'axes.linewidth': style['line_width'],
    'xtick.major.width': style['line_width'],
    'ytick.major.width': style['line_width'],
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': frame_style == "Full Box",
    'axes.spines.right': frame_style == "Full Box",
    'axes.spines.left': frame_style != "No Frame",
    'axes.spines.bottom': frame_style != "No Frame",
    'figure.facecolor': fig_bg,
    'axes.facecolor': axes_bg,
    'figure.dpi': export_dpi,
    'savefig.dpi': export_dpi,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ── Plot Generation Helper ──────────────────────────────────────────────────
def render_chart(ax, element, cond_data, chart_type, use_3d=False, is_combined=False):
    x = np.arange(len(samples))
    x_positions = np.arange(len(samples))
    
    # Prepare data for plotting
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
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    n_cond = len(detected_conditions)
    bar_w = 0.18
    offsets = np.arange(n_cond) * bar_w - (n_cond - 1) * bar_w / 2
    
    # ── Chart Type Rendering ─────────────────────────────────────────────
    if chart_type == "3D Bar" and use_3d:
        # Convert to 3D axis
        ax_3d = fig.add_subplot(111, projection='3d') if 'fig' in globals() else ax
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
            ax_3d.bar3d(xs, ys, zs, dx, dy, dz, color=custom_colors[cond], 
                       alpha=bar_alpha, edgecolor='white', linewidth=bar_edge_width)
        ax_3d.set_xticks(x_positions)
        ax_3d.set_xticklabels(sample_labels, fontsize=font_size_base)
        ax_3d.set_yticks(np.arange(n_cond))
        ax_3d.set_yticklabels(detected_conditions, fontsize=font_size_base-1)
        ax_3d.set_zlabel('Ion release (μg·cm⁻²)', fontsize=font_size_base+1)
        ax_3d.set_title(f'{element} ion release', fontsize=font_size_base+2, fontweight='bold' if title_bold else 'normal')
        ax_3d.view_init(elev=elev if use_3d else 30, azim=azim if use_3d else 45)
        return ax_3d
    
    elif chart_type == "Grouped Bar":
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            x_valid = x_positions[valid_mask] + offsets[i]
            bars = ax.bar(x_valid, vals[valid_mask], width=bar_w,
                         color=custom_colors[cond], label=cond, zorder=3,
                         edgecolor='white' if bar_edge_width > 0 else 'none',
                         linewidth=bar_edge_width, alpha=bar_alpha,
                         hatch=bar_hatch if bar_hatch != "none" else None)
            if show_error_bars:
                ax.errorbar(x_valid, vals[valid_mask], yerr=errs[valid_mask],
                           fmt='none', ecolor='#444', elinewidth=style['line_width'],
                           capsize=error_cap_size, capthick=style['line_width'], zorder=4)
            if show_data_labels:
                for j, (x_pos, val) in enumerate(zip(x_valid, vals[valid_mask])):
                    if not np.isnan(val) and val > 0:
                        ax.text(x_pos, val + 0.02, f'{val:.{label_precision}f}',
                               ha='center', va='bottom', fontsize=font_size_base-2, rotation=90)
    
    elif chart_type == "Stacked Bar":
        bottom = np.zeros(len(samples))
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            plot_vals = np.where(valid_mask, vals, 0)
            bars = ax.bar(x_positions, plot_vals, bottom=bottom,
                         color=custom_colors[cond], label=cond, zorder=3,
                         edgecolor='white' if bar_edge_width > 0 else 'none',
                         linewidth=bar_edge_width, alpha=bar_alpha,
                         hatch=bar_hatch if bar_hatch != "none" else None)
            bottom += plot_vals
            if show_error_bars and np.any(valid_mask):
                ax.errorbar(x_positions[valid_mask], bottom[valid_mask] - errs[valid_mask]/2,
                           yerr=errs[valid_mask], fmt='none', ecolor='#444',
                           elinewidth=style['line_width'], capsize=error_cap_size, zorder=4)
    
    elif chart_type == "Line Plot":
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            ax.plot(x_positions[valid_mask], vals[valid_mask],
                   marker=markers[i % len(markers)], markersize=style['marker_size'],
                   label=cond, color=custom_colors[cond], linewidth=style['line_width']*1.5)
            if show_error_bars:
                ax.errorbar(x_positions[valid_mask], vals[valid_mask], yerr=errs[valid_mask],
                           fmt='none', ecolor=custom_colors[cond], elinewidth=style['line_width'],
                           capsize=error_cap_size, alpha=0.8)
    
    elif chart_type == "Scatter Plot":
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            # Add slight jitter for better visibility
            jitter = np.random.normal(0, 0.03, size=np.sum(valid_mask))
            ax.scatter(x_positions[valid_mask] + jitter, vals[valid_mask],
                      marker=markers[i % len(markers)], s=style['marker_size']*15,
                      label=cond, color=custom_colors[cond], alpha=0.7, edgecolors='white', linewidth=0.5)
            if show_error_bars:
                ax.errorbar(x_positions[valid_mask], vals[valid_mask], yerr=errs[valid_mask],
                           fmt='none', ecolor=custom_colors[cond], elinewidth=style['line_width'],
                           capsize=error_cap_size, alpha=0.6)
    
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
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                           widths=0.6, showfliers=False)
            for patch, cond in zip(bp['boxes'], box_labels):
                patch.set_facecolor(custom_colors[cond])
                patch.set_alpha(bar_alpha)
            ax.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=font_size_base-1)
    
    elif chart_type == "Error Band":
        for i, cond in enumerate(detected_conditions):
            vals, errs = plot_data[cond]
            valid_mask = ~np.isnan(vals)
            if not np.any(valid_mask):
                continue
            x_valid = x_positions[valid_mask]
            ax.plot(x_valid, vals[valid_mask], label=cond, color=custom_colors[cond],
                   linewidth=style['line_width']*1.5)
            ax.fill_between(x_valid, 
                           np.maximum(0, vals[valid_mask] - errs[valid_mask]),
                           vals[valid_mask] + errs[valid_mask],
                           color=custom_colors[cond], alpha=0.2)
    
    # ── Axis Configuration ───────────────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sample_labels, fontsize=font_size_base)
    ax.set_ylabel('Ion release (μg·cm⁻²)', fontsize=font_size_base+1,
                 fontstyle='italic' if label_italic else 'normal')
    if not (chart_type == "3D Bar" and use_3d):
        ax.set_title(f'{element} ion release', fontsize=font_size_base+2,
                    fontweight='bold' if title_bold else 'normal', pad=12)
    
    # Y-axis limits and scale
    if not auto_ylim:
        ax.set_ylim(y_min, y_max)
    if y_log_scale and chart_type not in ["Box Plot"]:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    # ── Grid Configuration ───────────────────────────────────────────────
    if show_grid and frame_style != "No Frame":
        axis_map = {"Y-axis": "y", "X-axis": "x", "Both": "both"}
        ax.grid(True, axis=axis_map[grid_axis], linestyle=grid_style,
               linewidth=grid_width, alpha=grid_alpha, color=grid_color, zorder=0)
        if show_minor_grid:
            ax.minorticks_on()
            ax.grid(which='minor', axis=axis_map[grid_axis], linestyle=':',
                   linewidth=grid_width*0.5, alpha=grid_alpha*0.5, color=grid_color, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    
    # ── Frame Styling ────────────────────────────────────────────────────
    if frame_style != "No Frame":
        for spine_name, spine in ax.spines.items():
            if frame_style == "Full Box" or spine_name in ['left', 'bottom']:
                spine.set_visible(True)
                spine.set_linewidth(frame_width)
                spine.set_color(frame_color)
            else:
                spine.set_visible(False)
    
    # ── Legend Configuration ─────────────────────────────────────────────
    if show_legend and detected_conditions:
        handles = [mpatches.Patch(color=custom_colors[c], label=c) for c in detected_conditions]
        ax.legend(handles=handles, frameon=legend_frame, fontsize=font_size_base-1,
                 loc=legend_pos, ncol=legend_cols, handlelength=1.5, handleheight=1)
    
    # ── Annotation for Partial Detection ─────────────────────────────────
    if element in ('Al', 'Si'):
        ax.text(0.99, 0.97, '* n.d. = not detected', transform=ax.transAxes,
               fontsize=font_size_base-2, ha='right', va='top', 
               color='gray' if fig_bg == '#FFFFFF' else 'lightgray',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=axes_bg, alpha=0.7, edgecolor=frame_color))

# ── Streamlit Rendering ─────────────────────────────────────────────────────
elements_to_plot = list(data.keys())

if view_mode == "Individual Charts":
    for element in elements_to_plot:
        with st.expander(f"📊 {element} Ion Release", expanded=True):
            fig, ax = plt.subplots(figsize=(11, 6))
            if chart_type == "3D Bar" and use_3d:
                ax = fig.add_subplot(111, projection='3d')
            render_chart(ax, element, data[element], chart_type, use_3d, is_combined=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True, bbox_inches='tight')
            plt.close(fig)
else:
    st.subheader("📋 Combined Element Overview")
    if chart_type == "3D Bar" and use_3d:
        st.warning("3D charts work best in individual view. Rendering 2D fallback for grid view.")
        use_3d_grid = False
    else:
        use_3d_grid = use_3d
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    for idx, element in enumerate(elements_to_plot):
        ax = axes_flat[idx]
        if chart_type == "3D Bar" and use_3d_grid:
            ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        render_chart(ax, element, data[element], chart_type, use_3d_grid, is_combined=True)
    
    # Hide unused subplots
    for ax in axes_flat[len(elements_to_plot):]:
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True, bbox_inches='tight')
    plt.close(fig)

# ── Footer & Export Info ────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Elements", len(elements_to_plot))
with col2:
    st.metric("Samples", len(samples))
with col3:
    st.metric("Conditions", len(conditions))

st.caption(f"🎨 Style: {pub_style} | 📊 Type: {chart_type} | 🧊 3D: {'Yes' if use_3d else 'No'} | 📐 DPI: {export_dpi}")
st.success("✅ Charts rendered with publication-quality settings. Use browser print (Ctrl+P) or matplotlib toolbar to export.")
