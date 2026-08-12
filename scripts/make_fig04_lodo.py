"""Figure 4 - Cross-database transfer under leave-one-database-out evaluation.

Message: a binary magnetic label survives the change of DFT protocol with a
quantifiable penalty, whereas the moment magnitude does not.

Outputs: fig/fig_04.png, fig/fig_04.pdf, fig/fig_04_data.csv
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

MODEL = "lgbm"
C_IND, C_LODO = "#009E73", "#D55E00"          # Okabe-Ito green / vermillion
DBS = {"c2db": "C2DB\n(GPAW/PBE)", "jarvis2d": "JARVIS-2D\n(VASP/OptB88)",
       "twodmatpedia": "2DMatPedia\n(VASP/MP)"}

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
    raise SystemExit(f"[fig04] could not resolve {what or cands[0]!r}; tried {cands}; "
                     f"available columns: {list(df.columns)}")


def load(pattern):
    hits = sorted(OUT.glob(pattern))
    if not hits:
        raise SystemExit(f"[fig04] {pattern} not found in {OUT}")
    return pd.read_csv(hits[-1]), hits[-1].name


lodo, lodo_name = load("T4_lodo_results*.csv")
ind, ind_name = load("T4_indist_results*.csv")

for df in (lodo, ind):
    resolve(df, "scenario", what="scenario")
    resolve(df, "model", what="model")
f_col = resolve(lodo, "f1_macro", "f1", what="F1")
r_col = resolve(lodo, "r2", "R2", what="R2")
task_col = resolve(lodo, "task", what="task")


def series(df, scen, col, task=None):
    m = (df["scenario"] == scen) & (df["model"] == MODEL)
    if task is not None:
        m &= df[task_col] == task
    v = df.loc[m, col].dropna().to_numpy()
    if v.size == 0:
        raise SystemExit(f"[fig04] no rows for scenario={scen}, model={MODEL}"
                         + (f", task={task}" if task else "")
                         + f"; scenarios present: {sorted(set(df['scenario']))}")
    return v


# ------------------------------------------------------------------ plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})
rows, deltas = [], []
x = np.arange(len(DBS))
w = 0.34

for i, db in enumerate(DBS):
    a = series(ind, f"indist_{db}", f_col)
    b = series(lodo, f"LODO_{db}", f_col, task="T4cls")
    for pos, v, c, lab in ((i - w / 2, a, C_IND, "in-distribution"),
                           (i + w / 2, b, C_LODO, "LODO")):
        ax1.bar(pos, v.mean(), w, color=c, alpha=0.35, edgecolor=c, lw=0.9,
                hatch="" if lab == "in-distribution" else "//",
                label=lab if i == 0 else None)
        ax1.scatter(pos + RNG.uniform(-0.07, 0.07, v.size), v, s=11, color=c,
                    marker="o" if lab == "in-distribution" else "s",
                    linewidths=0, zorder=3)
        ax1.errorbar(pos, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor="black",
                     elinewidth=0.9, capsize=2.5, zorder=4)
        rows += [{"panel": "a", "db": db, "setting": lab, "seed_f1": s} for s in v]
    d = a.mean() - b.mean()
    ax1.annotate("", xy=(i + w / 2, b.mean()), xytext=(i + w / 2, a.mean()),
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    deltas.append((i, d))
    rows.append({"panel": "a_delta", "db": db, "delta_f1": d,
                 "n_indist": a.size, "n_lodo": b.size})

# one common height for all delta labels, clear of every bar and point
Y_DELTA = 0.965
for i, d in deltas:
    ax1.text(i + w / 2, Y_DELTA, rf"$\Delta$ = {d:.2f}", fontsize=7.5,
             ha="center", va="center")

ax1.axhline(0.5, color="#666666", ls=":", lw=0.8)
ax1.text(len(DBS) - 0.55, 0.508, "chance", fontsize=7, color="#666666")
ax1.set_xticks(x); ax1.set_xticklabels(DBS.values(), fontsize=7.5)
ax1.set_ylabel(r"F$_1^{\mathrm{macro}}$ (magnetic vs. nonmagnetic)")
ax1.set_ylim(0.4, 1.0)
ax1.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, alpha=0.3, lw=0.5); ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="lower left", ncol=2, columnspacing=1.4,
           handlelength=1.4, bbox_to_anchor=(-0.02, 1.00), fontsize=7.5)
ax1.text(0.5, -0.30, "(a)", transform=ax1.transAxes, ha="center", va="top", fontsize=9)

for i, db in enumerate(DBS):
    v = series(lodo, f"LODO_{db}", r_col, task="T4reg")
    ax2.bar(i, v.mean(), 0.5, color=C_LODO, alpha=0.35, edgecolor=C_LODO, lw=0.9,
            hatch="//")
    ax2.scatter(i + RNG.uniform(-0.09, 0.09, v.size), v, s=11, color=C_LODO,
                marker="s", linewidths=0, zorder=3)
    ax2.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor="black",
                 elinewidth=0.9, capsize=2.5, zorder=4)
    rows += [{"panel": "b", "db": db, "seed_r2": s} for s in v]

ax2.axhline(0, color="black", lw=0.9)
ax2.set_xticks(x); ax2.set_xticklabels(DBS.values(), fontsize=7.5)
ax2.set_ylabel(r"LODO $R^2$ ($\mu_\mathrm{B}$/atom)")
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, alpha=0.3, lw=0.5); ax2.set_axisbelow(True)
ax2.text(0.5, -0.30, "(b)", transform=ax2.transAxes, ha="center", va="top", fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "fig_04.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig_04.pdf", bbox_inches="tight", pad_inches=0.02)
pd.DataFrame(rows).to_csv(FIG / "fig_04_data.csv", index=False)
print(f"wrote fig/fig_04.{{png,pdf}} and fig_04_data.csv  "
      f"(sources: {ind_name}, {lodo_name}; model={MODEL})")
