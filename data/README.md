# Data access

This project does not redistribute any raw database files.

## C2DB (required for notebooks 01-05)

The Computational 2D Materials Database is developed and maintained by
DTU/CAMD and licensed under CC BY-NC 4.0. The bulk files are provided upon
request; see https://2dhub.org/c2db/c2db.html for instructions. After
obtaining them, place the following under `dataset/`:

- `c2db.db`      (ASE database snapshot)
- `c2db.tar.gz`  (per-material archive; required for the raw-record
                  cross-validation of the spin-spiral labels, which reads the
                  `results-asr.collect_spiral.json` files)

## Spin-spiral labels

The 164 ground-state labels (58 FM, 21 collinear AFM, 85 non-collinear, of
which 15 DM-driven spirals) are reconstructed by notebook 01 from the
published tables of:

> J. Sodequist and T. Olsen, npj Computational Materials 10, 170 (2024).

The reconstruction is validated in three independent ways, including a
row-by-row comparison against the raw archive records (147/164 covered, zero
ordering-vector discrepancies at 0.02 r.l.u. tolerance).

## JARVIS-2D and 2DMatPedia (notebook 06 only)

Both are fetched automatically through `jarvis-tools` (Figshare mirrors):
datasets `dft_2d` and `twod_matpd`. Downloads are cached under `dataset/` as
Parquet files. These sources are subject to their own licenses; see the
JARVIS and 2DMatPedia project pages.
