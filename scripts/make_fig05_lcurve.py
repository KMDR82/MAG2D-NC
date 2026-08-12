"""Figure 5 - Learning curves under group-aware subsampling.

Message: performance has not saturated at 164 materials, so the reported
ceilings are data-limited rather than representation-limited.

Outputs: fig/fig_05.png, fig/fig_05.pdf, fig/fig_05_data.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
ROOT = Path.home() / "MAG2D-NC"
OUT = ROOT / "output"
FIG = Path("fig"); FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

COLOR = {"logreg": "#0072B2", "lgbm": "#E69F00"}
MARKER = {"logreg": "o", "lgbm": "s"}
LINE = {"logreg": "-", "lgbm": "--"}
LABEL = {"logreg": "Logistic regression", "lgbm": "LightGBM"}

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
    raise SystemExit(f"[fig05] could not resolve {what or cands[0]!r}; tried {cands}; "
                     f"available columns: {list(df.columns)}")


hits = sorted(OUT.glob("T1_learning_curve*.csv"))
if not hits:
    raise SystemExit(f"[fig05] T1_learning_curve*.csv not found in {OUT}")
lc = pd.read_csv(hits[-1])

m_col = resolve(lc, "model", what="model")
x_col = resolve(lc, "fraction", "frac", "train_fraction", "pct", what="training fraction")
y_col = resolve(lc, "f1_macro", "f1", "f1_mean", what="F1")
per_fold = lc.groupby([m_col, x_col])[y_col].size().max() > 1

xs = np.sort(lc[x_col].unique())
xscale = 100.0 if xs.max() <= 1.001 else 1.0


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if v.size < 2:
        return v.mean(), v.mean()
    bs = np.random.default_rng(seed).choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])


# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(3.5, 3.0))
rows = []

for m in [k for k in ("logreg", "lgbm") if k in set(lc[m_col])]:
    sub = lc[lc[m_col] == m]
    mean, lo, hi = [], [], []
    for x in xs:
        v = sub.loc[sub[x_col] == x, y_col].to_numpy()
        mean.append(v.mean())
        if per_fold:
            a, b = boot_ci(v)
            lo.append(a); hi.append(b)
            ax.scatter(np.full(v.size, x * xscale) + RNG.uniform(-1.2, 1.2, v.size), v,
                       s=6, color=COLOR[m], alpha=0.30, linewidths=0, zorder=2,
                       rasterized=True)
            rows += [{"model": m, "fraction_pct": x * xscale, "fold_f1": s} for s in v]
        else:
            sd_col = next((c for c in ("f1_std", "std", "f1_sd") if c in lc.columns), None)
            if sd_col is None:
                raise SystemExit("[fig05] the learning-curve CSV holds one row per point "
                                 "and no dispersion column; uncertainty cannot be shown. "
                                 f"Columns: {list(lc.columns)}")
            s = sub.loc[sub[x_col] == x, sd_col].iloc[0]
            lo.append(v.mean() - s); hi.append(v.mean() + s)
            rows.append({"model": m, "fraction_pct": x * xscale, "mean_f1": v.mean(),
                         "dispersion": s, "dispersion_kind": sd_col})
    mean, lo, hi = map(np.asarray, (mean, lo, hi))
    dx = 0.9 if m == "lgbm" else -0.9          # tiny offset so bars do not overlap
    ax.errorbar(xs * xscale + dx, mean, yerr=[mean - lo, hi - mean], fmt=LINE[m],
                color=COLOR[m], marker=MARKER[m], ms=4, lw=1.3,
                elinewidth=0.9, capsize=2.5, zorder=4, label=LABEL[m])
    if per_fold:
        for x, a, b, mu in zip(xs, lo, hi, mean):
            rows.append({"model": m, "fraction_pct": x * xscale, "mean_f1": mu,
                         "ci_lo": a, "ci_hi": b})

ax.set_xlabel("training groups retained (%)")
ax.set_ylabel(r"F$_1^{\mathrm{macro}}$ (T1)")
ax.set_xticks(xs * xscale)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, lw=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.00), ncol=2,
          columnspacing=1.4, handlelength=1.8, fontsize=8)

fig.tight_layout()
fig.savefig(FIG / "fig_05.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig_05.pdf", bbox_inches="tight", pad_inches=0.02)
pd.DataFrame(rows).to_csv(FIG / "fig_05_data.csv", index=False)
print(f"wrote fig/fig_05.{{png,pdf}} and fig_05_data.csv  (source: {hits[-1].name}, "
      f"{'per-fold values with bootstrap CI' if per_fold else 'summary rows with stored dispersion'})")
