import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings

# Suppress non-critical matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Streamlit page configuration
st.set_page_config(page_title="Ion Release Analysis", layout="wide")
st.title("ICP-AES Ion Release Analysis Dashboard")

# ── Data (μg·cm⁻², converted from ppm × 10 mL / 10.2 cm²) ──────────────────
samples = ['CH0', 'CH45', 'PH0', 'PH45', 'CNH0', 'CNH45', 'PNH0', 'PNH45']

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
        'Ringer 7D': None,  # not detected
        'Ringer 1M': ([0.012, 0.019, 0.020, 0.021, 0.009, 0.010, 0.010, 0.020],
                      [0.001]*8),
        'LAC 7D':    ([0.033, 0.037, 0.043, 0.050, 0.022, 0.028, 0.028, 0.030],
                      [0.001]*8),
        'LAC 1M':    ([0.054, 0.068, 0.064, 0.071, 0.048, 0.054, 0.041, 0.048],
                      [0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]),
    },
    'Al': {
        'Ringer 7D': None,  # not detected
        'Ringer 1M': ([0.009, 0.010, 0.010, 0.015, None, None, None, None],
                      [0.001, 0.001, 0.001, 0.001, None, None, None, None]),
        'LAC 7D':    None,  # not detected
        'LAC 1M':    ([0.022, 0.025, 0.028, 0.030, 0.012, 0.015, 0.022, 0.027],
                      [0.001]*8),
    },
}

# ── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛠️ Chart Settings")

    font_size = st.slider("🔤 Font Size", min_value=8, max_value=16, value=11, step=1)

    st.subheader("🎨 Condition Colors")
    default_colors = {
        'Ringer 7D': '#85B7EB',
        'Ringer 1M': '#185FA5',
        'LAC 7D':    '#F0997B',
        'LAC 1M':    '#D85A30',
    }
    user_colors = {}
    for cond, default in default_colors.items():
        user_colors[cond] = st.color_picker(cond, default)

    view_mode = st.radio("📊 View Mode", ["Individual", "Combined (2x3 Grid)"])

    st.subheader("🖼️ Background")
    fig_bg_choice = st.radio("Figure Background", ["White", "Transparent", "Custom"])
    if fig_bg_choice == "Custom":
        fig_bg = st.color_picker("Custom Figure Color", "#F5F5F5")
    elif fig_bg_choice == "White":
        fig_bg = "#FFFFFF"
    else:
        fig_bg = None

    axes_bg = st.color_picker("Axes Plot Background", "#FFFFFF")

    st.subheader("📜 Legend")
    show_legend = st.checkbox("Show Legend", value=True)
    legend_pos = st.selectbox(
        "Legend Position",
        ["upper left", "upper right", "lower left", "lower right", "center right", "best"],
        disabled=not show_legend
    )

# ── Apply Matplotlib Params ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
    'font.size':      font_size,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

conditions = list(user_colors.keys())
n = len(samples)
n_cond = len(conditions)
bar_w = 0.18
offsets = np.arange(n_cond) * bar_w - (n_cond - 1) * bar_w / 2

# ── Plot Generation Helper ──────────────────────────────────────────────────
def render_element_on_axis(ax, element, cond_data, is_combined=False):
    x = np.arange(n)
    for i, cond in enumerate(conditions):
        vals_errs = cond_data[cond]
        if vals_errs is None:
            continue

        vals, errs = vals_errs
        plot_vals = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        plot_errs = np.array([e if e is not None else np.nan for e in errs], dtype=float)
        valid_mask = ~np.isnan(plot_vals)
        x_valid = x[valid_mask] + offsets[i]

        ax.bar(x_valid, plot_vals[valid_mask], width=bar_w,
               color=user_colors[cond], label=cond, zorder=3,
               edgecolor='white', linewidth=0.4)
        ax.errorbar(x_valid, plot_vals[valid_mask], yerr=plot_errs[valid_mask],
                    fmt='none', ecolor='#444', elinewidth=0.8, capsize=3,
                    capthick=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(samples, fontsize=font_size)
    ax.set_ylabel('Ion release (μg·cm⁻²)', fontsize=font_size+1)
    ax.set_title(f'{element} ion release', fontsize=font_size+2, fontweight='normal', pad=8)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(axes_bg)

    detected = [c for c in conditions if cond_data[c] is not None]
    if show_legend and detected:
        handles = [mpatches.Patch(color=user_colors[c], label=c) for c in detected]
        ncol = 1 if is_combined else 2
        ax.legend(handles=handles, frameon=False, fontsize=font_size-1,
                  loc=legend_pos, ncol=ncol)

    if element in ('Al', 'Si'):
        ax.text(0.99, 0.97, '* n.d. = not detected in that condition',
                transform=ax.transAxes, fontsize=font_size-2,
                ha='right', va='top', color='gray')

# ── Streamlit Rendering ─────────────────────────────────────────────────────
if view_mode == "Individual":
    for element, cond_data in data.items():
        st.subheader(f"{element} Ion Release")
        fig, ax = plt.subplots(figsize=(10, 5))
        if fig_bg:
            fig.patch.set_facecolor(fig_bg)
        render_element_on_axis(ax, element, cond_data, is_combined=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
else:
    # Combined 2x3 grid
    st.subheader("Combined Element Overview")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    if fig_bg:
        fig.patch.set_facecolor(fig_bg)
    axes_flat = axes.flatten()
    elements = list(data.keys())
    for idx, (element, cond_data) in enumerate(data.items()):
        render_element_on_axis(axes_flat[idx], element, cond_data, is_combined=True)
    # Hide any leftover axes if dataset changes size
    for ax in axes_flat[len(elements):]:
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.success("✅ Charts rendered successfully with current settings.")
