"""Regenerate the manuscript figures from the result tables.

Reads CSV/JSON files from results/ and writes PDF+PNG pairs to figs/.
Run from the repository root:

    python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("results")
OUT = Path("figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GREEN, BLUE, GRAY, ORANGE = "#2c6e49", "#7e9dc8", "#9e9e9e", "#e07b39"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def fig2_t1():
    """T1 scores across model families with the permutation-null band."""
    summ = pd.read_csv(RES / "T1_summary.csv")
    perm = json.loads((RES / "T1_permutation.json").read_text())
    order = ["dummy_majority", "dummy_stratified", "cgcnn", "alignn_lite",
             "lgbm", "logreg"]
    labels = ["Majority\ndummy", "Stratified\ndummy", "CGCNN", "ALIGNN-lite",
              "LightGBM", "Logistic\nregr."]
    colors = [GRAY, GRAY, BLUE, BLUE, GREEN, GREEN]
    rows = summ.set_index("model").reindex(order)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(order))
    ax.bar(x, rows["f1_mean"],
           yerr=[rows["f1_mean"] - rows["ci_lo"], rows["ci_hi"] - rows["f1_mean"]],
           capsize=3, color=colors, width=0.62)
    null_mean = perm["logreg"]["null_mean"]
    null_p95 = perm["logreg"]["null_p95"]
    ax.axhspan(null_mean, null_p95, color="orange", alpha=0.15, lw=0)
    ax.axhline(null_mean, color="darkorange", ls="--", lw=1)
    ax.axhline(null_p95, color="darkorange", ls=":", lw=1)
    ax.text(0.02, null_p95 + 0.004, "permutation null: mean / 95th pct.",
            color="darkorange", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"F$_1^{\rm macro}$ (T1)")
    ax.set_ylim(0.3, 0.68)
    _save(fig, "fig2_t1")


def fig3_shap(top=15):
    """Top descriptors by out-of-fold mean |SHAP|."""
    imp = pd.read_csv(RES / "T1_importances.csv").nlargest(top, "shap_mean_abs")
    feats = imp["feature"].tolist()
    vals = imp["shap_mean_abs"].tolist()
    cols = ["#c0392b" if f.startswith(("sym_", "soc_")) else "#7f8c8d"
            for f in feats]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    yp = np.arange(len(feats))[::-1]
    ax.barh(yp, vals, color=cols, height=0.7)
    nice = [f.replace("comp_", "").replace("sym_", "sym: ").replace("_", " ")
            for f in feats]
    ax.set_yticks(yp)
    ax.set_yticklabels(nice, fontsize=7)
    ax.set_xlabel("mean |SHAP| (out-of-fold)")
    _save(fig, "fig3_shap")


def fig4_lodo():
    """Cross-database transfer: in-distribution vs LODO, and LODO R2."""
    lodo = pd.read_csv(RES / "T4_lodo_results.csv")
    ind = pd.read_csv(RES / "T4_indist_results.csv")
    dbs = ["c2db", "jarvis2d", "twodmatpedia"]
    names = ["C2DB", "JARVIS-2D", "2DMatPedia"]
    f1_ind = [ind[(ind.scenario == f"indist_{d}") & (ind.model == "lgbm")]
              ["f1_macro"].mean() for d in dbs]
    f1_lodo = [lodo[(lodo.scenario == f"LODO_{d}") & (lodo.model == "lgbm")
                    & (lodo.task == "T4cls")]["f1_macro"].mean() for d in dbs]
    r2_lodo = [lodo[(lodo.scenario == f"LODO_{d}") & (lodo.model == "lgbm")
                    & (lodo.task == "T4reg")]["r2"].mean() for d in dbs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.7))
    x = np.arange(3)
    w = 0.36
    a1.bar(x - w / 2, f1_ind, w, label="in-dist", color=GREEN)
    a1.bar(x + w / 2, f1_lodo, w, label="LODO", color=ORANGE)
    for i, (a, b) in enumerate(zip(f1_ind, f1_lodo)):
        a1.annotate("", xy=(i + w / 2, b), xytext=(i + w / 2, a),
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
        a1.text(i + w / 2 + 0.06, (a + b) / 2, f"\u2212{a - b:.2f}", fontsize=7)
    a1.axhline(0.5, color="gray", ls=":", lw=1)
    a1.text(2.0, 0.505, "dummy", fontsize=6.5, color="gray")
    a1.set_xticks(x)
    a1.set_xticklabels(names, fontsize=7.5)
    a1.set_ylabel(r"F$_1^{\rm macro}$ (magnetic cls.)")
    a1.set_ylim(0.4, 1.0)
    a1.legend(fontsize=7)
    a2.bar(x, r2_lodo, color=[GREEN, ORANGE, ORANGE], width=0.5)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(x)
    a2.set_xticklabels(names, fontsize=7.5)
    a2.set_ylabel(r"LODO $R^2$ ($\mu_B$/atom)")
    _save(fig, "fig4_lodo")


def fig5_lcurve():
    """Learning curves under group-aware subsampling."""
    lc = pd.read_csv(RES / "T1_learning_curve.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for model, style, color in [("logreg", "o-", GREEN), ("lgbm", "s--", BLUE)]:
        sub = lc[lc.model == model].sort_values("fraction")
        ax.errorbar(sub["fraction"] * 100, sub["f1_mean"], yerr=sub["f1_std"],
                    fmt=style, capsize=3, color=color,
                    label={"logreg": "logistic", "lgbm": "LightGBM"}[model])
    ax.set_xlabel("% of training groups")
    ax.set_ylabel(r"F$_1^{\rm macro}$")
    ax.legend(fontsize=8)
    _save(fig, "fig5_lcurve")


if __name__ == "__main__":
    for fn in (fig2_t1, fig3_shap, fig4_lodo, fig5_lcurve):
        try:
            fn()
            print(f"{fn.__name__}: written")
        except FileNotFoundError as exc:
            print(f"{fn.__name__}: skipped ({exc})")
