# =============================================================================
# cell_E_graph_neighbour_stats.py
#
# PURPOSE (Solid State Communications, Reviewer 1, remark (b))
#   "Why exactly 12 nearest neighbours / this cut-off radius, given the
#    diversity of the crystal structures?"
#
#   This cell answers that question with numbers computed from the 164
#   structures actually used by the graph models, using the SAME neighbour
#   construction as notebook 05 (ASE neighbour list, 8 A cut-off, atoms'
#   own periodic boundary conditions).
#
#   For every atom of every structure it computes
#     n_cut      : number of neighbours inside the 8 A cut-off
#     d_min      : nearest-neighbour distance
#     n_shell1   : first-coordination-shell size, d <= SHELL_TOL * d_min
#     r_12       : distance to the 12th nearest neighbour
#   and aggregates them per material and per crystal system.
#
#   The claim the Methods sentence will make is only defensible if the data
#   support it, so the cell prints the two decisive quantities explicitly:
#     max(n_shell1)         -> is 12 always >= the full first shell?
#     max(r_12) vs 8 A      -> is the cut-off ever the active constraint?
#   Nothing is asserted here; the printed numbers decide the wording.
#
# OUTPUTS
#   output/graph_neighbour_statistics.csv        per-material summary (164 rows)
#   output/graph_neighbour_by_crystal_system.csv per-crystal-system summary
#   fig/fig_S1_neighbours.png / .pdf             SI figure (300 dpi + vector)
#   fig/fig_S1_neighbours_data.csv               per-atom table behind the figure
#
# RUNTIME  ~1-2 minutes on CPU (symmetry analysis dominates).
# =============================================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from ase.io import read as ase_read
from ase.neighborlist import neighbor_list

# -----------------------------------------------------------------------------
# CONFIG  (must mirror the graph construction of notebook 05)
# -----------------------------------------------------------------------------
ROOT       = Path.home() / "MAG2D-NC"
OUT        = ROOT / "output"
FIG        = ROOT / "fig"
STRUCT_DIR = ROOT / "structures_164"

CUTOFF     = 8.0    # A   -- nb05 CONFIG: graph cut-off radius
MAX_NBR    = 12     #     -- nb05 CONFIG: nearest-neighbour cap per atom
SHELL_TOL  = 1.20   #     -- first coordination shell: d <= SHELL_TOL * d_min
SYMPREC    = 0.1    # A   -- symmetry tolerance for the crystal-system label
N_EXPECTED = 164

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# transition metals, used only for an auxiliary magnetic-sublattice summary
TM = set(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Hf Ta W Re Os Ir Pt Au Hg".split()
)

# -----------------------------------------------------------------------------
# 1. locate structures
# -----------------------------------------------------------------------------
if not STRUCT_DIR.is_dir():
    for cand in [ROOT / "dataset" / "structures_164", ROOT / "output" / "structures_164"]:
        if cand.is_dir():
            STRUCT_DIR = cand
            break
files = sorted(STRUCT_DIR.glob("*.extxyz")) or sorted(STRUCT_DIR.glob("*.xyz"))
assert files, f"no structure files found under {STRUCT_DIR}"
print(f"structures: {len(files)} in {STRUCT_DIR}")
if len(files) != N_EXPECTED:
    warnings.warn(f"expected {N_EXPECTED} structures, found {len(files)}")

# -----------------------------------------------------------------------------
# 2. crystal system (spglib via pymatgen; falls back to 'unknown')
# -----------------------------------------------------------------------------
try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    HAVE_PMG = True
except Exception as exc:                                      # pragma: no cover
    print(f"pymatgen unavailable ({exc}); crystal systems will be 'unknown'")
    HAVE_PMG = False


def crystal_system(atoms):
    if not HAVE_PMG:
        return "unknown", np.nan
    try:
        st = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(st, symprec=SYMPREC)
        return sga.get_crystal_system(), int(sga.get_space_group_number())
    except Exception:
        return "unknown", np.nan


# -----------------------------------------------------------------------------
# 3. per-atom neighbour statistics
# -----------------------------------------------------------------------------
atom_rows = []
mat_rows = []
thin_cells = []

for f in files:
    at = ase_read(f)
    mid = f.stem
    cs, sgn = crystal_system(at)
    a_len, b_len, c_len = at.cell.lengths()

    # vacuum sanity: with pbc along the out-of-plane axis, c must exceed 2*cutoff
    if bool(at.pbc[2]) and c_len < 2 * CUTOFF:
        thin_cells.append((mid, round(float(c_len), 2)))

    i, j, d = neighbor_list("ijd", at, CUTOFF)
    nat = len(at)
    symbols = at.get_chemical_symbols()

    for a in range(nat):
        da = np.sort(d[i == a])
        if da.size == 0:                       # isolated atom inside the cut-off
            atom_rows.append(dict(
                material=mid, crystal_system=cs, space_group=sgn,
                atom_index=a, element=symbols[a], is_tm=symbols[a] in TM,
                n_cut=0, d_min=np.nan, n_shell1=0, r_shell1=np.nan, r_12=np.nan,
                c_len=c_len,
            ))
            continue
        d_min = float(da[0])
        r_shell1 = SHELL_TOL * d_min
        n_shell1 = int((da <= r_shell1).sum())
        r_12 = float(da[MAX_NBR - 1]) if da.size >= MAX_NBR else np.nan
        atom_rows.append(dict(
            material=mid, crystal_system=cs, space_group=sgn,
            atom_index=a, element=symbols[a], is_tm=symbols[a] in TM,
            n_cut=int(da.size), d_min=d_min, n_shell1=n_shell1,
            r_shell1=r_shell1, r_12=r_12, c_len=c_len,
        ))

atoms_df = pd.DataFrame(atom_rows)
print(f"atoms analysed: {len(atoms_df)}  "
      f"({atoms_df['material'].nunique()} materials, "
      f"{atoms_df['crystal_system'].nunique()} crystal systems)")
if thin_cells:
    print(f"WARNING cells with periodic c < 2*cutoff: {thin_cells}")

# -----------------------------------------------------------------------------
# 4. per-material and per-crystal-system aggregates
# -----------------------------------------------------------------------------
mat = (atoms_df.groupby(["material", "crystal_system", "space_group"], dropna=False)
       .agg(n_atoms=("atom_index", "size"),
            n_cut_min=("n_cut", "min"),
            n_cut_median=("n_cut", "median"),
            n_cut_max=("n_cut", "max"),
            d_min_min=("d_min", "min"),
            n_shell1_min=("n_shell1", "min"),
            n_shell1_max=("n_shell1", "max"),
            r_shell1_max=("r_shell1", "max"),
            r_12_max=("r_12", "max"),
            atoms_below_12=("n_cut", lambda s: int((s < MAX_NBR).sum())))
       .reset_index())
mat.to_csv(OUT / "graph_neighbour_statistics.csv", index=False)

by_cs = (atoms_df.groupby("crystal_system", dropna=False)
         .agg(n_materials=("material", "nunique"),
              n_atoms=("atom_index", "size"),
              n_cut_median=("n_cut", "median"),
              n_cut_min=("n_cut", "min"),
              d_min_median=("d_min", "median"),
              n_shell1_median=("n_shell1", "median"),
              n_shell1_max=("n_shell1", "max"),
              r_shell1_median=("r_shell1", "median"),
              r_12_median=("r_12", "median"),
              r_12_max=("r_12", "max"))
         .reset_index()
         .sort_values("n_materials", ascending=False))
by_cs.to_csv(OUT / "graph_neighbour_by_crystal_system.csv", index=False)

# -----------------------------------------------------------------------------
# 5. the two decisive numbers for the Methods sentence
# -----------------------------------------------------------------------------
n_shell1_max = int(atoms_df["n_shell1"].max())
n_shell1_p99 = float(atoms_df["n_shell1"].quantile(0.99))
frac_shell_covered = float((atoms_df["n_shell1"] <= MAX_NBR).mean())
r12_max = float(np.nanmax(atoms_df["r_12"].values))
r12_median = float(np.nanmedian(atoms_df["r_12"].values))
atoms_lt_12 = int((atoms_df["n_cut"] < MAX_NBR).sum())
n_cut_min = int(atoms_df["n_cut"].min())
n_cut_median = float(atoms_df["n_cut"].median())

print("\n=== neighbour / cut-off justification ===")
print(f"neighbours inside {CUTOFF:.0f} A per atom : "
      f"median {n_cut_median:.0f}, min {n_cut_min}, max {int(atoms_df['n_cut'].max())}")
print(f"atoms with fewer than {MAX_NBR} neighbours inside the cut-off : {atoms_lt_12}")
print(f"first-shell size (d <= {SHELL_TOL}*d_min) : "
      f"median {atoms_df['n_shell1'].median():.0f}, "
      f"99th pct {n_shell1_p99:.0f}, max {n_shell1_max}")
print(f"fraction of atoms whose FULL first shell fits in {MAX_NBR} neighbours : "
      f"{frac_shell_covered:.4f}")
print(f"distance to the {MAX_NBR}th neighbour : "
      f"median {r12_median:.2f} A, max {r12_max:.2f} A  (cut-off {CUTOFF:.0f} A)")
print("\nper crystal system:")
print(by_cs.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

tm = atoms_df[atoms_df["is_tm"]]
if len(tm):
    print(f"\ntransition-metal sites only (n={len(tm)}): "
          f"first-shell median {tm['n_shell1'].median():.0f}, max {int(tm['n_shell1'].max())}; "
          f"r_12 median {np.nanmedian(tm['r_12']):.2f} A")

print(f"\nwrote {OUT/'graph_neighbour_statistics.csv'}")
print(f"wrote {OUT/'graph_neighbour_by_crystal_system.csv'}")

# -----------------------------------------------------------------------------
# 6. SI figure
# -----------------------------------------------------------------------------
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
})
OKABE = {
    "cut":    "#0072B2",   # neighbours inside the cut-off
    "shell":  "#D55E00",   # first coordination shell
    "r12":    "#009E73",   # distance to the 12th neighbour
    "ref":    "#555555",
}
PANEL_Y = -0.30            # identical in every panel

order = [c for c in by_cs["crystal_system"].tolist()]
short = {"triclinic": "tri", "monoclinic": "mono", "orthorhombic": "ortho",
         "tetragonal": "tetra", "trigonal": "trig", "hexagonal": "hex",
         "cubic": "cubic", "unknown": "unk"}
labels = [short.get(c, str(c)[:5]) for c in order]

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.85))

# (a) neighbours inside the cut-off, all atoms, with the 12-neighbour cap marked
ax = axes[0]
counts = atoms_df["n_cut"].values
bins = np.arange(counts.min() - 0.5, counts.max() + 1.5, max(1, (counts.max() - counts.min()) // 40) or 1)
ax.hist(counts, bins=bins, color=OKABE["cut"], edgecolor="white", linewidth=0.3, zorder=3)
ax.axvline(MAX_NBR, color=OKABE["ref"], linestyle="--", linewidth=1.0, zorder=4)
ax.annotate(f"cap = {MAX_NBR}", xy=(MAX_NBR, ax.get_ylim()[1] * 0.92),
            xytext=(4, 0), textcoords="offset points", fontsize=8,
            color=OKABE["ref"], ha="left", va="top")
ax.set_xlabel(f"neighbours within {CUTOFF:.0f} " + r"$\mathrm{\AA}$")
ax.set_ylabel("atoms")

# (b) distance to the 12th neighbour vs the first-shell radius, per crystal system
ax = axes[1]
r12_groups = [atoms_df.loc[atoms_df["crystal_system"] == c, "r_12"].dropna().values for c in order]
sh_groups = [atoms_df.loc[atoms_df["crystal_system"] == c, "r_shell1"].dropna().values for c in order]
pos = np.arange(len(order))
bp1 = ax.boxplot(r12_groups, positions=pos - 0.18, widths=0.3, patch_artist=True,
                 showfliers=False, zorder=3)
bp2 = ax.boxplot(sh_groups, positions=pos + 0.18, widths=0.3, patch_artist=True,
                 showfliers=False, zorder=3)
for bp, col, hatch in ((bp1, OKABE["r12"], ""), (bp2, OKABE["shell"], "///")):
    for box in bp["boxes"]:
        box.set(facecolor=col, alpha=0.75, edgecolor="black", linewidth=0.6, hatch=hatch)
    for key in ("medians", "whiskers", "caps"):
        for el in bp[key]:
            el.set(color="black", linewidth=0.6)
ax.axhline(CUTOFF, color=OKABE["ref"], linestyle=":", linewidth=1.0, zorder=2)
ax.annotate(f"cut-off {CUTOFF:.0f} " + r"$\mathrm{\AA}$", xy=(len(order) - 0.6, CUTOFF),
            xytext=(0, 2), textcoords="offset points", fontsize=8,
            color=OKABE["ref"], ha="right", va="bottom")
ax.set_xticks(pos)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel(r"distance ($\mathrm{\AA}$)")
ax.set_xlabel("crystal system")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=OKABE["r12"], alpha=0.75,
                         edgecolor="black", linewidth=0.6, label=f"{MAX_NBR}th neighbour"),
           plt.Rectangle((0, 0), 1, 1, facecolor=OKABE["shell"], alpha=0.75, hatch="///",
                         edgecolor="black", linewidth=0.6,
                         label=f"first shell ({SHELL_TOL:g}" + r"$d_\mathrm{min}$)")]
ax.legend(handles=handles, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=2, handlelength=1.2, columnspacing=1.0)

# (c) first-shell size per crystal system, with the 12-neighbour cap marked
ax = axes[2]
sh_n = [atoms_df.loc[atoms_df["crystal_system"] == c, "n_shell1"].values for c in order]
bp3 = ax.boxplot(sh_n, positions=pos, widths=0.5, patch_artist=True, showfliers=True, zorder=3)
for box in bp3["boxes"]:
    box.set(facecolor=OKABE["shell"], alpha=0.75, edgecolor="black", linewidth=0.6, hatch="///")
for key in ("medians", "whiskers", "caps"):
    for el in bp3[key]:
        el.set(color="black", linewidth=0.6)
for fl in bp3["fliers"]:
    fl.set(marker="o", markersize=2.0, markerfacecolor="none",
           markeredgecolor="black", markeredgewidth=0.5)
ax.axhline(MAX_NBR, color=OKABE["ref"], linestyle="--", linewidth=1.0, zorder=4)
ax.annotate(f"cap = {MAX_NBR}", xy=(len(order) - 0.6, MAX_NBR), xytext=(0, 2),
            textcoords="offset points", fontsize=8, color=OKABE["ref"],
            ha="right", va="bottom")
ax.set_xticks(pos)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("first-shell coordination number")
ax.set_xlabel("crystal system")

for ax, tag in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.5, PANEL_Y, tag, transform=ax.transAxes,
            ha="center", va="top", fontsize=9)

fig.tight_layout(w_pad=1.4)
fig.savefig(FIG / "fig_S1_neighbours.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig_S1_neighbours.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

atoms_df.to_csv(FIG / "fig_S1_neighbours_data.csv", index=False)
print(f"wrote {FIG/'fig_S1_neighbours.png'} / .pdf")
print(f"wrote {FIG/'fig_S1_neighbours_data.csv'} ({len(atoms_df)} atom rows)")
