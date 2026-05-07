import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings

# Suppress non-critical matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Streamlit page configuration
st.set_page_config(page_title="Ion Release Analysis", layout="wide")
st.title("ICP-AES Ion Release Analysis")

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

# ── Style ────────────────────────────────────────────────────────────────────
colors = {
    'Ringer 7D': '#85B7EB',
    'Ringer 1M': '#185FA5',
    'LAC 7D':    '#F0997B',
    'LAC 1M':    '#D85A30',
}

conditions = list(colors.keys())
n = len(samples)
n_cond = len(conditions)
bar_w = 0.18
offsets = np.arange(n_cond) * bar_w - (n_cond - 1) * bar_w / 2

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
    'font.size':      11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Streamlit Layout & Plot Generation ───────────────────────────────────────
for element, cond_data in data.items():
    st.subheader(f"{element} Ion Release")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)

    for i, cond in enumerate(conditions):
        vals_errs = cond_data[cond]
        if vals_errs is None:
            continue  # not detected — skip this condition

        vals, errs = vals_errs

        # Convert to float arrays, replacing None with np.nan for safe plotting
        plot_vals = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        plot_errs = np.array([e if e is not None else np.nan for e in errs], dtype=float)

        # Mask to only plot valid (non-nan) data points
        valid_mask = ~np.isnan(plot_vals)
        x_valid = x[valid_mask] + offsets[i]

        ax.bar(
            x_valid,
            plot_vals[valid_mask],
            width=bar_w,
            color=colors[cond],
            label=cond,
            zorder=3,
            edgecolor='white',
            linewidth=0.4,
        )
        ax.errorbar(
            x_valid,
            plot_vals[valid_mask],
            yerr=plot_errs[valid_mask],
            fmt='none',
            ecolor='#444',
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(samples, fontsize=11)
    ax.set_ylabel('Ion release (μg·cm⁻²)', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # legend (only conditions that have data for this element)
    detected = [c for c in conditions if cond_data[c] is not None]
    if detected:
        handles = [mpatches.Patch(color=colors[c], label=c) for c in detected]
        ax.legend(handles=handles, frameon=False, fontsize=10,
                  loc='upper left', ncol=2)

    # note for partial detection
    if element in ('Al', 'Si'):
        ax.text(0.99, 0.97,
                '* n.d. = not detected in that condition',
                transform=ax.transAxes, fontsize=9,
                ha='right', va='top', color='gray')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # Crucial for Streamlit to prevent memory leaks

st.success("✅ All charts rendered successfully.")
