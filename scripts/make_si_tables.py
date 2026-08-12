"""Regenerate every numeric table of the Supplementary Material from the run
records, and print them as LaTeX. Nothing is typed by hand: if a printed value
disagrees with supplementary.tex, the document is wrong, not this script.

Usage:  python scripts/make_si_tables.py            # print all tables
        python scripts/make_si_tables.py t1 t4      # print selected tables
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "MAG2D-NC"
OUT = ROOT / "output"
CKPT = ROOT / "checkpoints"


def latest(pattern, base=OUT):
    hits = sorted(base.glob(pattern))
    if not hits:
        print(f"  !! missing: {pattern}")
        return None
    return hits[-1]


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    bs = np.random.default_rng(seed).choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])


def rule(title):
    print(f"\n%% ---------- {title} ----------")


# --------------------------------------------------------------- S2: T1
def table_t1():
    rule("Table S2: T1 metrics")
    f = latest("runs.csv") or latest("T1_summary*.csv")
    if f is None:
        return
    df = pd.read_csv(f)
    if "task" in df.columns:
        df = df[df["task"].astype(str).str.upper() == "T1"]
    g = latest("progress.csv", CKPT / "T1_gnn")
    if g is not None:
        gd = pd.read_csv(g).rename(columns={"rep": "repeat"})
        df = pd.concat([df, gd], ignore_index=True)
    for m, sub in df.groupby("model"):
        v = sub["f1_macro"].to_numpy()
        lo, hi = boot_ci(v)
        mcc = sub["mcc"].mean() if "mcc" in sub.columns else np.nan
        bal = sub["bal_acc"].mean() if "bal_acc" in sub.columns else np.nan
        print(f"{m:18s} & ${v.mean():.3f}$ $[{lo:.3f},{hi:.3f}]$ & "
              f"${mcc:.3f}$ & ${bal:.3f}$ \\\\   % n={v.size}")


# ------------------------------------------------------- S2: permutation
def table_perm():
    rule("Table S4: permutation test")
    f = latest("T1_permutation*.json")
    if f is None:
        return
    d = json.loads(f.read_text())
    print(f"%% source: {f.name}")
    for m, r in d.items():
        print(f"{m:18s} & ${r['observed']:.4f}$ & ${r['null_mean']:.4f}$ & "
              f"${r['null_p95']:.4f}$ & ${r['p_value']:.3f}$ \\\\")


# ---------------------------------------------------------- S3: ablation
def table_a1():
    rule("Table S5: descriptor-group ablation")
    f = latest("T1_ablation_A1*.csv")
    if f is None:
        return
    df = pd.read_csv(f)
    key = "variant" if "variant" in df.columns else df.columns[0]
    print(df.pivot_table(index=key, columns="model", values="f1_macro").round(3).to_string())


# ------------------------------------------------------ S4: learning curve
def table_a3():
    rule("Table S6: learning curves")
    f = latest("T1_learning_curve*.csv")
    if f is None:
        return
    df = pd.read_csv(f)
    xcol = "fraction" if "fraction" in df.columns else df.columns[1]
    agg = df.groupby(["model", xcol])["f1_macro"].agg(["mean", "std", "size"])
    print(agg.round(3).to_string())


# --------------------------------------------------------------- S5: cost
def table_cost():
    rule("Table S7: graph-model cost")
    f = latest("progress.csv", CKPT / "T1_gnn")
    if f is None:
        return
    df = pd.read_csv(f)
    for m, sub in df.groupby("model"):
        v = sub["f1_macro"].to_numpy()
        print(f"{m:14s} & ${v.mean():.3f} \\pm {v.std(ddof=1):.3f}$ & "
              f"${sub['sec'].median():.0f}$\\,s & ${sub['vram_mb'].max():.0f}$\\,MB "
              f"\\\\   % n={v.size}")


# ----------------------------------------------------------------- S6: T2
def table_t2():
    rule("Table S8/S9: four-class task")
    f = latest("T2_results*.csv")
    if f is not None:
        df = pd.read_csv(f)
        print(df.groupby("model")[["f1_macro", "mcc"]].agg(["mean", "std"]).round(3).to_string())
    r = latest("T2_perclass_recall*.json")
    if r is None:
        print("  !! per-class recall JSON missing; rerun the T2 cell to write it")
        return
    d = json.loads(r.read_text())
    rng = np.random.default_rng(0)
    for c, rec in d.items():
        v = np.asarray(rec["values"], float)
        bs = rng.choice(v, size=(5000, v.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{c:24s} & ${v.mean():.3f}$ & $[{lo:.3f}, {hi:.3f}]$ \\\\   % n={v.size}")


# ----------------------------------------------------------------- S7: T3
def table_t3():
    rule("Table S10: anisotropy regression")
    f = latest("T3_results*.csv")
    if f is None:
        return
    df = pd.read_csv(f)
    print(df.groupby(["target", "model"])[["mae", "r2"]].mean().round(4).to_string())


# ----------------------------------------------------------------- S8: T4
def table_t4():
    rule("Tables S11-S13: cross-database transfer")
    lo_f, in_f = latest("T4_lodo_results*.csv"), latest("T4_indist_results*.csv")
    if lo_f is None or in_f is None:
        return
    lo, ind = pd.read_csv(lo_f), pd.read_csv(in_f)
    cls = lo[lo.task == "T4cls"] if "task" in lo.columns else lo
    print("-- classification (in-dist vs LODO) --")
    a = ind.groupby(["scenario", "model"])["f1_macro"].agg(["mean", "std"]).round(3)
    b = cls.groupby(["scenario", "model"])["f1_macro"].agg(["mean", "std"]).round(3)
    print(a.to_string()); print(b.to_string())
    if "task" in lo.columns:
        print("-- regression (LODO) --")
        print(lo[lo.task == "T4reg"].groupby(["scenario", "model"])[["mae_uB", "r2"]]
              .mean().round(3).to_string())
    ov = latest("T4_overlap_stats*.csv")
    if ov is not None:
        print("-- overlap --"); print(pd.read_csv(ov).to_string(index=False))


TABLES = {"t1": table_t1, "perm": table_perm, "a1": table_a1, "a3": table_a3,
          "cost": table_cost, "t2": table_t2, "t3": table_t3, "t4": table_t4}

def main(argv=None):
    # ignore anything that is not a known table name: inside Jupyter, sys.argv
    # carries the kernel connection file rather than user arguments
    argv = sys.argv[1:] if argv is None else argv
    wanted = [a for a in argv if a in TABLES] or list(TABLES)
    for key in wanted:
        TABLES[key]()


if __name__ == "__main__":
    main()
    print("\n%% compare each value against supplementary.tex before submitting")
