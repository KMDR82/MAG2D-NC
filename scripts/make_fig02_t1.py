"""Figure 2 - T1 primary task: fold-level performance and paired improvement.

Message: composition and symmetry descriptors separate collinear from
non-collinear order in the large majority of folds, but by a modest margin.

Run from anywhere:  python make_fig02_t1.py
Outputs: fig/fig_02.png, fig/fig_02.pdf, fig/fig_02_data.csv
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
ROOT = Path.home() / "MAG2D-NC"
OUT = ROOT / "output"
FIG = Path("fig"); FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)          # jitter only

COLOR = {"dummy_majority": "#999999", "dummy_stratified": "#999999",
         "dummy": "#999999", "logreg": "#0072B2", "lgbm": "#E69F00",
         "cgcnn": "#009E73", "alignn_lite": "#CC79A7"}
MARKER = {"dummy_majority": "x", "dummy_stratified": "x", "dummy": "x",
          "logreg": "o", "lgbm": "s", "cgcnn": "^", "alignn_lite": "D"}
LABEL = {"dummy_majority": "Majority\ndummy", "dummy_stratified": "Stratified\ndummy",
         "dummy": "Stratified\ndummy", "logreg": "Logistic\nregression",
         "lgbm": "LightGBM", "cgcnn": "CGCNN", "alignn_lite": "ALIGNN-lite"}

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
    raise SystemExit(
        f"[fig02] could not resolve {what or cands[0]!r}; tried {cands}; "
        f"available columns: {list(df.columns)}")


def load_csv(*patterns):
    for pat in patterns:
        hits = sorted(OUT.glob(pat)) or sorted(ROOT.rglob(pat))
        if hits:
            return pd.read_csv(hits[-1]), hits[-1]
    raise SystemExit(f"[fig02] none of {patterns} found under {ROOT}")


# ------------------------------------------------------------------ data
runs, runs_path = load_csv("runs.csv", "T1_folds*.csv", "T1_summary*.csv")
if "task" in runs.columns:
    runs = runs[runs["task"].astype(str).str.upper() == "T1"]
m_col = resolve(runs, "model", what="model")
f_col = resolve(runs, "f1_macro", "f1", "f1_mean", what="fold F1")
tab = runs[[m_col, f_col]].rename(columns={m_col: "model", f_col: "f1_macro"})

gnn_hits = sorted((ROOT / "checkpoints").rglob("progress.csv"))
gnn_hits = [p for p in gnn_hits if "SMOKE" not in str(p)]
if gnn_hits:
    g = pd.read_csv(gnn_hits[-1])
    gm = resolve(g, "model", what="model")
    gf = resolve(g, "f1_macro", "f1", what="fold F1")
    tab = pd.concat([tab, g[[gm, gf]].rename(columns={gm: "model", gf: "f1_macro"})],
                    ignore_index=True)

order = [m for m in ["dummy_majority", "dummy_stratified", "dummy", "cgcnn",
                     "alignn_lite", "lgbm", "logreg"] if m in set(tab["model"])]
if not order:
    raise SystemExit(f"[fig02] no known model names in {sorted(set(tab['model']))}")


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    bs = np.random.default_rng(seed).choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])


perm_files = sorted(OUT.glob("T1_permutation*.json"))
if not perm_files:
    raise SystemExit(f"[fig02] T1_permutation*.json not found in {OUT}")
perm_path = perm_files[-1]           # latest timestamp, not the first
perm = json.loads(perm_path.read_text())
if len(perm_files) > 1:
    print(f"[fig02] {len(perm_files)} permutation files present; using the latest: "
          f"{perm_path.name}")

# integrity gate: the stored 'observed' score must match the fold means we plot
for _m in ("logreg", "lgbm"):
    if _m in perm and "observed" in perm[_m] and _m in set(tab["model"]):
        _mean = tab.loc[tab.model == _m, "f1_macro"].mean()
        if abs(perm[_m]["observed"] - _mean) > 0.01:
            raise SystemExit(
                f"[fig02] {perm_path.name} reports observed F1 {perm[_m]['observed']:.3f} "
                f"for {_m}, but the fold data give {_mean:.3f}. These come from different "
                "runs - resolve before plotting.")

# panel (b): paired per-fold improvement over the stratified dummy.
BASE = "dummy_stratified" if "dummy_stratified" in set(tab["model"]) else "dummy"
key_cols = [c for c in ("seed", "repeat", "rep", "fold") if c in runs.columns]
if not key_cols:
    raise SystemExit(f"[fig02] no fold identifier columns in {runs_path.name}; "
                     f"available: {list(runs.columns)}")
piv = (runs.pivot_table(index=key_cols, columns=m_col, values=f_col)
       .dropna(subset=["logreg", BASE]))
diff = (piv["logreg"] - piv[BASE]).to_numpy()
if diff.size == 0:
    raise SystemExit("[fig02] no paired folds for logreg vs the stratified dummy")
d_lo, d_hi = boot_ci(diff)

# ------------------------------------------------------------------ plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.3),
                               gridspec_kw={"width_ratios": [1.35, 1]})

rows = []
for i, m in enumerate(order):
    v = tab.loc[tab.model == m, "f1_macro"].to_numpy()
    lo, hi = boot_ci(v)
    c = COLOR.get(m, "#444444")
    bp = ax1.boxplot(v, positions=[i], widths=0.55, showfliers=False,
                     patch_artist=True, medianprops=dict(color="black", lw=1.0),
                     whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                     boxprops=dict(lw=0.7))
    bp["boxes"][0].set(facecolor=c, alpha=0.28, edgecolor=c)
    ax1.scatter(i + RNG.uniform(-0.16, 0.16, v.size), v, s=5, alpha=0.45,
                color=c, marker=MARKER.get(m, "o"), linewidths=0, zorder=3,
                rasterized=True)
    ax1.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none",
                 ecolor="black", elinewidth=1.1, capsize=3, zorder=4)
    ax1.scatter([i], [v.mean()], s=22, color="black", marker="_", zorder=5)
    rows += [{"panel": "a", "model": m, "fold_f1": x} for x in v]
    rows.append({"panel": "a_summary", "model": m, "mean": v.mean(),
                 "ci_lo": lo, "ci_hi": hi, "n_folds": v.size})

ax1.set_xticks(range(len(order)))
ax1.set_xticklabels([LABEL.get(m, m).replace("\n", " ") for m in order],
                    fontsize=7.5, rotation=28, ha="right",
                    rotation_mode="anchor")
ax1.set_ylabel(r"F$_1^{\mathrm{macro}}$ (T1, per fold)")
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, alpha=0.3, lw=0.5)
ax1.set_axisbelow(True)
ax1.text(0.5, -0.34, "(a)", transform=ax1.transAxes, ha="center", va="top", fontsize=9)

counts, edges = np.histogram(diff, bins=22)
h = counts.max()
ax2.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
        color="#0072B2", alpha=0.45, edgecolor="white", lw=0.4, zorder=2)

# reserved annotation band on top: reference lines stop below it, so no overlap
TOP = h * 1.62
ax2.set_ylim(0, TOP)
ax2.vlines(0, 0, h * 1.06, color="black", lw=1.0, zorder=3)
ax2.vlines(diff.mean(), 0, h * 1.06, color="#0072B2", lw=1.6, zorder=3)
ax2.hlines(h * 1.02, d_lo, d_hi, color="#0072B2", lw=1.0, zorder=3)
ax2.plot([d_lo, d_hi], [h * 1.02] * 2, "|", color="#0072B2", ms=4, zorder=3)

won = (diff > 0).mean() * 100
ax2.text(0.03, 0.985,
         rf"mean $\Delta$ = {diff.mean():+.3f}  [95% CI {d_lo:+.3f}, {d_hi:+.3f}]",
         transform=ax2.transAxes, fontsize=7.5, color="#0072B2", va="top", ha="left")
ax2.text(0.03, 0.895,
         f"ahead of the dummy in {won:.0f}% of folds\n"
         rf"permutation $p$ = {perm['logreg']['p_value']:.3f} ($n$ = 1000)",
         transform=ax2.transAxes, fontsize=7.5, color="black", va="top", ha="left")

ax2.set_xlabel(r"per-fold $\Delta$F$_1^{\mathrm{macro}}$ (logistic $-$ stratified dummy)")
ax2.set_ylabel(f"folds (n = {diff.size})")
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, alpha=0.3, lw=0.5)
ax2.set_axisbelow(True)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax2.set_yticks([t for t in ax2.get_yticks() if 0 <= t <= h * 1.06])
ax2.text(0.5, -0.34, "(b)", transform=ax2.transAxes, ha="center", va="top", fontsize=9)

rows += [{"panel": "b", "fold_delta_f1": float(x)} for x in diff]
rows.append({"panel": "b_summary", "mean_delta": float(diff.mean()),
             "ci_lo": float(d_lo), "ci_hi": float(d_hi),
             "n_folds": int(diff.size),
             "perm_p_logreg": perm["logreg"]["p_value"],
             "perm_p_lgbm": perm["lgbm"]["p_value"]})

fig.tight_layout()
fig.savefig(FIG / "fig_02.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig_02.pdf", bbox_inches="tight", pad_inches=0.02)
pd.DataFrame(rows).to_csv(FIG / "fig_02_data.csv", index=False)
print("wrote fig/fig_02.{png,pdf} and fig_02_data.csv")
print(f"  sources: {runs_path.name}"
      + (f", {gnn_hits[-1].parent.name}/progress.csv" if gnn_hits else "")
      + f", {perm_path.name}")
print(f"  panel (b): {diff.size} paired folds, mean delta {diff.mean():+.3f} "
      f"[{d_lo:+.3f}, {d_hi:+.3f}]")
