"""Figure 3 - Interpretability: leading descriptors and cross-method agreement.

Message: the inversion-symmetry flag leads both attribution methods, whose
rankings agree only moderately, while the ligand-SOC proxies carry no weight.

Outputs: fig/fig_03.png, fig/fig_03.pdf, fig/fig_03_data.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# ----------------------------------------------------------------- config
ROOT = Path.home() / "MAG2D-NC"
OUT = ROOT / "output"
FIG = Path("fig"); FIG.mkdir(exist_ok=True)
TOP = 15

GROUPS = ["symmetry", "SOC proxy", "composition"]
COLOR = {"symmetry": "#D55E00", "SOC proxy": "#0072B2", "composition": "#8C8C8C"}
HATCH = {"symmetry": "///", "SOC proxy": "...", "composition": ""}
MARKER = {"symmetry": "D", "SOC proxy": "^", "composition": "o"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
})


def resolve(df, *cands, what=""):
    for c in cands:
        if c in df.columns:
            return c
    raise SystemExit(f"[fig03] could not resolve {what or cands[0]!r}; tried {cands}; "
                     f"available columns: {list(df.columns)}")


# ------------------------------------------------------------------ data
hits = sorted(OUT.glob("T1_importances*.csv"))
if not hits:
    raise SystemExit(f"[fig03] T1_importances*.csv not found in {OUT}")
imp = pd.read_csv(hits[-1])

f_col = resolve(imp, "feature", "name", what="feature name")
s_col = resolve(imp, "shap_mean_abs", "shap", "mean_abs_shap", "shap_importance",
                what="mean |SHAP|")
c_col = resolve(imp, "coef_abs", "abs_coef", "coef_mean_abs", "coef", "std_coef",
                what="standardized coefficient magnitude")
sd_col = next((c for c in ("shap_std", "shap_sd", "shap_std_across_repeats")
               if c in imp.columns), None)

imp["group"] = np.where(imp[f_col].str.startswith("sym_"), "symmetry",
                np.where(imp[f_col].str.startswith("soc_"), "SOC proxy", "composition"))
imp["shap_rank"] = imp[s_col].abs().rank(ascending=False)
imp["coef_rank"] = imp[c_col].abs().rank(ascending=False)
rho, pval = spearmanr(imp["shap_rank"], imp["coef_rank"])
n_feat = len(imp)
top = imp.nlargest(TOP, s_col).iloc[::-1].reset_index(drop=True)

# ------------------------------------------------------------------ plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.6),
                               gridspec_kw={"width_ratios": [1.45, 1],
                                            "wspace": 0.38})

# ---- panel (a): ranked bars, key inside the reserved right margin ----
y = np.arange(len(top))
err = top[sd_col].to_numpy() if sd_col else None
ax1.barh(y, top[s_col], height=0.72,
         color=[COLOR[g] for g in top["group"]],
         hatch=[HATCH[g] for g in top["group"]],
         edgecolor="white", linewidth=0.6,
         xerr=err, error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2))
nice = (top[f_col].str.replace("comp_", "", regex=False)
        .str.replace("sym_", "sym: ", regex=False)
        .str.replace("soc_", "soc: ", regex=False)
        .str.replace("_", " ", regex=False))
ax1.set_yticks(y)
ax1.set_yticklabels(nice, fontsize=7.5)
ax1.set_ylim(-0.8, len(top) - 0.2)
ax1.set_xlim(0, float(top[s_col].max()) * 1.55)     # free margin for the key
ax1.set_xlabel("mean |SHAP| (out-of-fold)")
ax1.spines[["top", "right"]].set_visible(False)
ax1.xaxis.grid(True, alpha=0.3, lw=0.5)
ax1.set_axisbelow(True)

shown = [g for g in GROUPS if g in set(top["group"])]   # only groups present here
bar_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[g], hatch=HATCH[g],
                             edgecolor="white") for g in shown]
ax1.legend(bar_handles, shown, frameon=False, loc="lower right",
           bbox_to_anchor=(1.0, 0.02), handlelength=1.6, fontsize=7.5)

soc = imp[imp["group"] == "SOC proxy"]
if len(soc):
    ax1.text(0.99, 0.42,
             f"SOC proxies rank\n{int(soc['shap_rank'].min())}\u2013"
             f"{int(soc['shap_rank'].max())} of {n_feat}",
             transform=ax1.transAxes, ha="right", va="center", fontsize=7.5,
             color=COLOR["SOC proxy"], linespacing=1.35)

ax1.text(0.5, -0.18, "(a)", transform=ax1.transAxes, ha="center", va="top",
         fontsize=9)

# ---- panel (b): rank agreement, key on the strip above its own axes ----
lim = (0, n_feat + 3)
ax2.plot(lim, lim, ls="--", lw=0.7, color="#666666", zorder=1)
for g in GROUPS:
    sub = imp[imp["group"] == g]
    ax2.scatter(sub["shap_rank"], sub["coef_rank"], s=15, color=COLOR[g],
                marker=MARKER[g], alpha=0.8, linewidths=0, label=g,
                zorder=3, rasterized=True)

lead = imp.nsmallest(1, "shap_rank").iloc[0]
ax2.scatter([lead["shap_rank"]], [lead["coef_rank"]], s=110, facecolors="none",
            edgecolors=COLOR["symmetry"], linewidths=1.2, zorder=5)

ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xticks([0, 25, 50, 75, 100, 125])
ax2.set_yticks([0, 25, 50, 75, 100, 125])
ax2.set_xlabel("rank by mean |SHAP|")
ax2.set_ylabel("rank by |standardized coefficient|")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(True, alpha=0.3, lw=0.5)
ax2.set_axisbelow(True)

mark_handles = [Line2D([], [], color=COLOR[g], marker=MARKER[g], ls="none", ms=4)
                for g in GROUPS]
ax2.legend(mark_handles, GROUPS, frameon=False, loc="lower left",
           bbox_to_anchor=(-0.02, 1.00), ncol=3, columnspacing=1.1,
           handletextpad=0.3, fontsize=7.5)

ax2.text(0.5, -0.18, "(b)", transform=ax2.transAxes, ha="center", va="top",
         fontsize=9)

# ---------------------------------------------------------------- output
fig.tight_layout()
fig.savefig(FIG / "fig_03.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig_03.pdf", bbox_inches="tight", pad_inches=0.02)

out = imp[[f_col, s_col, c_col, "group", "shap_rank", "coef_rank"]].copy()
out["in_panel_a_top15"] = out[f_col].isin(top[f_col])
if sd_col:
    out["shap_std"] = imp[sd_col]
out.to_csv(FIG / "fig_03_data.csv", index=False)

print(f"wrote fig/fig_03.{{png,pdf}} and fig_03_data.csv  (source: {hits[-1].name}, "
      f"n_features={n_feat}, cross-method Spearman rho={rho:.3f}, p={pval:.2g}"
      + (f", SHAP error bars from {sd_col}" if sd_col else ", no SHAP dispersion column")
      + ")")
print("  note: rho goes in the caption, not inside the axes")
