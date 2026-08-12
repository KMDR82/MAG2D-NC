"""Regenerate all manuscript figures.

This is a thin driver: each figure lives in its own script so that it can be
run and reviewed independently.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py 02 04      # selected figures

Figure 1 (the workflow schematic) is drawn by hand in diagrams.net; its source
is figs/fig1_workflow.drawio.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "02": "make_fig02_t1.py",          # T1 fold distributions + paired improvement
    "03": "make_fig03_interpret.py",   # SHAP ranking + cross-method agreement
    "04": "make_fig04_lodo.py",        # cross-database transfer
    "05": "make_fig05_lcurve.py",      # learning curves
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    wanted = [a for a in argv if a in SCRIPTS] or list(SCRIPTS)
    for key in wanted:
        path = HERE / SCRIPTS[key]
        if not path.exists():
            print(f"[fig {key}] missing script: {path.name}")
            continue
        print(f"\n=== figure {key}: {path.name} ===")
        saved = sys.argv
        try:
            sys.argv = [str(path)]           # keep child scripts free of our args
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:            # scripts stop deliberately on bad input
            print(f"[fig {key}] stopped: {exc}")
        finally:
            sys.argv = saved


if __name__ == "__main__":
    main()
