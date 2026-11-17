# =============================================================================
# make_fig_S1_neighbours.py
#
# Redraws SI Figure S1 from the per-atom table written by
# cell_E_graph_neighbour_stats.py. No recomputation: every number comes from
# fig/fig_S1_neighbours_data.csv.
#
# Fixes relative to the first draft:
#   - panel labels moved clear of the rotated tick labels (same offset in all
#     three panels, so they stay aligned)
#   - head-room above the reference lines in panels (b) and (c)
#   - coarser, parity-free bins in panel (a)
#   - rotation_mode="anchor" so tick labels sit under their boxes
#
# Run:  python make_fig_S1_neighbours.py
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path.home() / "MAG2D-NC"
FIG = ROOT / "fig"

CUTOFF = 8.0
MAX_NBR = 12
SHELL_TOL = 1.20
PANEL_Y = -0.42          # identical in every panel

df = pd.read_csv(FIG / "fig_S1_neighbours_data.csv")
print(f"read {len(df)} atom rows, {df['material'].nunique()} materials")

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
})
OKABE = {"cut": "#0072B2", "shell": "#D55E00", "r12": "#009E73", "ref": "#555555"}

order = (df.groupby("crystal_system")["material"].nunique()
           .sort_values(ascending=False).index.tolist())
short = {"triclinic": "tri", "monoclinic": "mono", "orthorhombic": "ortho",
         "tetragonal": "tetra", "trigonal": "trig", "hexagonal": "hex",
         "cubic": "cubic", "unknown": "unk"}
labels = [short.get(c, str(c)[:5]) for c in order]
pos = np.arange(len(order))

fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.05))

# --- (a) neighbours inside the cut-off, all atoms -----------------------------
ax = axes[0]
counts = df["n_cut"].values
lo = 5 * int(np.floor(counts.min() / 5))
hi = 5 * int(np.ceil(counts.max() / 5)) + 5
ax.hist(counts, bins=np.arange(lo, hi + 1, 5), color=OKABE["cut"],
        edgecolor="white", linewidth=0.4, zorder=3)
ax.axvline(MAX_NBR, color=OKABE["ref"], linestyle="--", linewidth=1.0, zorder=4)
ax.set_xlim(0, hi)
ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
ax.annotate(f"cap = {MAX_NBR}", xy=(MAX_NBR, 0.96),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=OKABE["ref"], ha="left", va="top")
ax.set_xlabel(f"neighbors within {CUTOFF:.0f} " + r"$\mathrm{\AA}$")
ax.set_ylabel("atoms")

# --- (b) 12th-neighbor distance vs first-shell radius ------------------------
ax = axes[1]
r12 = [df.loc[df["crystal_system"] == c, "r_12"].dropna().values for c in order]
rsh = [df.loc[df["crystal_system"] == c, "r_shell1"].dropna().values for c in order]
bp1 = ax.boxplot(r12, positions=pos - 0.18, widths=0.30, patch_artist=True,
                 showfliers=False, zorder=3)
bp2 = ax.boxplot(rsh, positions=pos + 0.18, widths=0.30, patch_artist=True,
                 showfliers=False, zorder=3)
for bp, col, hatch in ((bp1, OKABE["r12"], ""), (bp2, OKABE["shell"], "///")):
    for box in bp["boxes"]:
        box.set(facecolor=col, alpha=0.78, edgecolor="black", linewidth=0.6, hatch=hatch)
    for key in ("medians", "whiskers", "caps"):
        for el in bp[key]:
            el.set(color="black", linewidth=0.6)
ax.axhline(CUTOFF, color=OKABE["ref"], linestyle=":", linewidth=1.0, zorder=2)
ax.set_ylim(1.6, 9.4)
ax.text(0.98, 0.90, f"cut-off {CUTOFF:.0f} " + r"$\mathrm{\AA}$",
        transform=ax.transAxes, fontsize=8, color=OKABE["ref"], ha="right", va="bottom")
ax.set_xticks(pos)
ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylabel(r"distance ($\mathrm{\AA}$)")
ax.set_xlabel("crystal system")
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=OKABE["r12"], alpha=0.78,
                  edgecolor="black", linewidth=0.6, label=f"{MAX_NBR}th neighbor"),
    plt.Rectangle((0, 0), 1, 1, facecolor=OKABE["shell"], alpha=0.78, hatch="///",
                  edgecolor="black", linewidth=0.6,
                  label=f"first shell ({SHELL_TOL:g}" + r"$d_\mathrm{min}$)"),
]
ax.legend(handles=handles, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=2, handlelength=1.1, columnspacing=0.9)

# --- (c) first-shell coordination number --------------------------------------
ax = axes[2]
shn = [df.loc[df["crystal_system"] == c, "n_shell1"].values for c in order]
bp3 = ax.boxplot(shn, positions=pos, widths=0.5, patch_artist=True,
                 showfliers=True, zorder=3)
for box in bp3["boxes"]:
    box.set(facecolor=OKABE["shell"], alpha=0.78, edgecolor="black",
            linewidth=0.6, hatch="///")
for key in ("medians", "whiskers", "caps"):
    for el in bp3[key]:
        el.set(color="black", linewidth=0.6)
for fl in bp3["fliers"]:
    fl.set(marker="o", markersize=2.0, markerfacecolor="none",
           markeredgecolor="black", markeredgewidth=0.5)
ax.axhline(MAX_NBR, color=OKABE["ref"], linestyle="--", linewidth=1.0, zorder=4)
ax.set_ylim(0, 14.5)
ax.text(0.98, 0.88, f"cap = {MAX_NBR}", transform=ax.transAxes, fontsize=8,
        color=OKABE["ref"], ha="right", va="bottom")
ax.set_xticks(pos)
ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylabel("first-shell coordination number")
ax.set_xlabel("crystal system")

for ax, tag in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.5, PANEL_Y, tag, transform=ax.transAxes,
            ha="center", va="top", fontsize=9)

fig.tight_layout(w_pad=1.6)
fig.subplots_adjust(bottom=0.30)
fig.savefig(FIG / "fig_S1_neighbours.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig_S1_neighbours.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"wrote {FIG/'fig_S1_neighbours.png'} / .pdf")
