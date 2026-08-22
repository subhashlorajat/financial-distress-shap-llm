"""
Run the complete project notebook pipeline, in order, across all members.

    python run_notebooks.py                  every notebook, in numeric order
    python run_notebooks.py --list           show what would run, run nothing
    python run_notebooks.py --from 02        start at notebook 02
    python run_notebooks.py --only 03 04     just those two
    python run_notebooks.py --regenerate-splits   rebuild splits.json (rarely wanted)

Notebooks are discovered by numeric prefix in notebooks/, so ordering is
implicit and adding a stage requires no change here:

    01_data_exploration.ipynb      Member 1  data pipeline + EDA
    02_data_splitting.ipynb        Member 2  Step 4
    03_baseline_models.ipynb       Member 2  Step 5
    04_tabtransformer.ipynb        Member 2  Step 6
    05_model_comparison.ipynb      Member 2  comparison + hand-off
    06_shap_explainability.ipynb   Member 3  Step 7   <- drops in automatically
    07_llm_narratives.ipynb        Member 3  Step 8
    08_narrative_evaluation.ipynb  Member 3  Step 9

Each notebook is executed in place, so outputs, tables and figures are stored
in the .ipynb itself. That is what makes the submitted notebooks self-evidencing
rather than just source listings.

IMPORTANT - data/splits.json
    The split is immutable once committed: every number in the paper depends on
    it. This runner exports H9DLGA_SKIP_PREP=1, which notebook 02 honours by
    reusing the existing file. Pass --regenerate-splits only if you intend to
    invalidate and re-run every downstream result.

H9DLGA Group Project - National College of Ireland.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except ImportError:
    sys.exit("Missing dependencies. Run:\n"
             "    pip install nbformat nbclient nbconvert ipykernel")

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"

# Notebooks may train models or call an LLM API, so the ceiling is generous.
CELL_TIMEOUT = 7200          # seconds per cell
KERNEL = "python3"

OWNER = {"01": "Member 1  (Data Engineer)",
         "02": "Member 2  (Model Engineer)",
         "03": "Member 2  (Model Engineer)",
         "04": "Member 2  (Model Engineer)",
         "05": "Member 2  (Model Engineer)",
         "06": "Member 3  (Explainability)",
         "07": "Member 3  (Explainability)",
         "08": "Member 3  (Explainability)"}


def discover() -> list[Path]:
    """All notebooks with a numeric prefix, in order. Checkpoints ignored."""
    found = [p for p in NOTEBOOK_DIR.glob("*.ipynb")
             if re.match(r"^\d+[_-]", p.name)
             and ".ipynb_checkpoints" not in str(p)]
    return sorted(found, key=lambda p: p.name)


def prefix(path: Path) -> str:
    m = re.match(r"^(\d+)", path.name)
    return m.group(1) if m else "??"


def owner(path: Path) -> str:
    return OWNER.get(prefix(path), "unassigned")


def execute(path: Path) -> float:
    """Run one notebook in place. Raises on the first failing cell."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    start = time.perf_counter()
    try:
        client.execute()
    finally:
        # Persist whatever completed, so a failure still leaves partial evidence.
        nbformat.write(nb, path)
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Execute the project notebooks in order.")
    ap.add_argument("--list", action="store_true",
                    help="show the execution plan and exit")
    ap.add_argument("--from", dest="start", metavar="NN",
                    help="start from this notebook prefix, e.g. 02")
    ap.add_argument("--to", dest="end", metavar="NN",
                    help="stop after this notebook prefix")
    ap.add_argument("--only", nargs="+", metavar="NN",
                    help="run only these prefixes, e.g. --only 03 04")
    ap.add_argument("--regenerate-splits", action="store_true",
                    help="allow notebook 02 to rebuild data/splits.json "
                         "(invalidates every downstream result)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a failing notebook")
    args = ap.parse_args()

    notebooks = discover()
    if not notebooks:
        sys.exit(f"No numbered notebooks found in {NOTEBOOK_DIR}")

    if args.only:
        wanted = {p.zfill(2) for p in args.only}
        notebooks = [n for n in notebooks if prefix(n) in wanted]
    else:
        if args.start:
            notebooks = [n for n in notebooks if prefix(n) >= args.start.zfill(2)]
        if args.end:
            notebooks = [n for n in notebooks if prefix(n) <= args.end.zfill(2)]

    if not notebooks:
        sys.exit("Selection matched no notebooks.")

    # Guard the immutable split unless explicitly overridden.
    if args.regenerate_splits:
        os.environ.pop("H9DLGA_SKIP_PREP", None)
    else:
        os.environ["H9DLGA_SKIP_PREP"] = "1"

    print("=" * 70)
    print("H9DLGA - FULL NOTEBOOK PIPELINE")
    print("Explainable Financial Health Scoring")
    print("=" * 70)
    print(f"\nnotebooks/  {NOTEBOOK_DIR}")
    print(f"splits.json {'WILL BE REGENERATED' if args.regenerate_splits else 'preserved (reused)'}")
    print(f"\nExecution plan ({len(notebooks)} notebooks):")
    for n in notebooks:
        print(f"  {prefix(n)}  {n.name:<34} {owner(n)}")

    if args.list:
        return

    print()
    results, total = [], 0.0
    for i, nb_path in enumerate(notebooks, 1):
        print("\n" + "#" * 70)
        print(f"# [{i}/{len(notebooks)}]  {nb_path.name}")
        print(f"# {owner(nb_path)}")
        print("#" * 70)
        try:
            secs = execute(nb_path)
            total += secs
            results.append((nb_path.name, "OK", secs))
            print(f"\n  completed in {secs:.1f}s")
        except CellExecutionError as exc:
            results.append((nb_path.name, "FAILED", 0.0))
            print(f"\n  FAILED: {str(exc).splitlines()[-1][:200]}")
            if not args.continue_on_error:
                print("\nStopping. Fix the cell above, then re-run with:")
                print(f"    python run_notebooks.py --from {prefix(nb_path)}")
                _summary(results, total)
                sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            results.append((nb_path.name, "ERROR", 0.0))
            print(f"\n  ERROR: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                _summary(results, total)
                sys.exit(1)

    _summary(results, total)


def _summary(results, total: float) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status, secs in results:
        mark = "ok   " if status == "OK" else "FAIL "
        print(f"  {mark} {name:<36} {secs:7.1f}s" if status == "OK"
              else f"  {mark} {name:<36}       -")
    ok = sum(1 for _, s, _ in results if s == "OK")
    print(f"\n  {ok}/{len(results)} succeeded | total {total/60:.1f} min")

    if ok == len(results):
        print("\nAll notebooks executed. Outputs are saved inside each .ipynb.")


if __name__ == "__main__":
    main()
