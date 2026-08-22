"""
Run the complete Member 2 pipeline end to end.

    python run_all_member_2.py              full run (Optuna tuning, 5 seeds per model)
    python run_all_member_2.py --quick      smoke test, a few minutes

Stages
    1  data preparation and company-level splitting
    2  baselines            Logistic Regression, XGBoost
    3  neural models        MLP, TabTransformer
    4  comparison report    tables and figures
    5  hand-off validation  confirms Member 3 can load everything

H9DLGA Group Project - Member 2 (Model Engineer) (Smit).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STAGES = [
    ("Data preparation and splitting", "data_prep.py", []),
    ("Baseline models", "train_baselines.py", ["--trials", "40"]),
    ("Neural models", "train_neural.py", ["--trials", "25"]),
    ("Comparison report", "build_report.py", []),
    ("Hand-off validation", "validate_handoff.py", []),
]


def run_stage(title: str, script: str, args: list[str], quick: bool) -> float:
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)

    cmd = [sys.executable, script] + (["--quick"] if quick and args else args)
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=SRC)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\nFAILED: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[{title}: {elapsed:.1f}s]")
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="few tuning trials, single seed - for smoke testing")
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse the existing splits.json (recommended once committed)")
    args = ap.parse_args()

    print("=" * 70)
    print("H9DLGA - MEMBER 2 (MODEL ENGINEER) PIPELINE")
    print("Explainable Financial Health Scoring")
    print("=" * 70)
    if args.quick:
        print("\nMODE: quick smoke test - results are NOT publication grade")

    total = 0.0
    for title, script, extra in STAGES:
        if args.skip_prep and script == "data_prep.py":
            print(f"\n(skipping {script} - reusing existing splits)")
            continue
        total += run_stage(title, script, extra, args.quick)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {total/60:.1f} minutes")
    print("=" * 70)
    print("\nDeliverables:")
    print("  results/model_comparison.csv        full metric table")
    print("  results/model_comparison_paper.csv  paper-ready table")
    print("  results/figures/                    5 figures")
    print("  models/                             checkpoints + preprocessing artefacts")
    print("\nMember 3 entry point:  from predict import explain_ready")


if __name__ == "__main__":
    main()
