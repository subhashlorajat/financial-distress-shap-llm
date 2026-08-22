"""
Step 5 - Baseline classifiers: Logistic Regression and XGBoost.

Logistic Regression establishes the performance floor. If a deep model cannot
beat a linear one, something is wrong with the setup rather than with the
architecture.

XGBoost is the serious baseline. Gradient boosting is the reigning default on
tabular data, so RQ1 ("do deep tabular models outperform gradient boosting?")
is only meaningful if XGBoost is properly tuned rather than left at defaults.

Run:  python src/train_baselines.py [--trials 40] [--quick]

H9DLGA Group Project - Member 2 (Model Engineer).
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
from data_prep import load_prepared
from evaluation import (aggregate_seeds, class_weights, evaluate,
                        log_experiment, print_report, timer)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------------------------
def train_logistic(Xtr, ytr, Xva, yva, Xte, yte, seed=C.PRIMARY_SEED):
    """Linear floor. Features are standardised because L2 regularisation
    penalises raw coefficient magnitude, which is scale-dependent."""
    scaler = StandardScaler().fit(Xtr)          # fitted on TRAIN only
    with timer() as t:
        model = LogisticRegression(max_iter=5000, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
        model.fit(scaler.transform(Xtr), ytr)
    train_s = t()

    res = {}
    for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        Xs = scaler.transform(X)
        res[name] = evaluate(y, model.predict(Xs), model.predict_proba(Xs))
    n_params = int(model.coef_.size + model.intercept_.size)
    return model, scaler, res, train_s, n_params


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------
def _xgb_params(trial, seed):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "random_state": seed,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "tree_method": "hist",
    }


def tune_xgboost(Xtr, ytr, Xva, yva, n_trials=C.N_TRIALS_XGB, seed=C.PRIMARY_SEED):
    """Bayesian search, scored on the VALIDATION fold only.

    The test set is never touched here. Tuning against test would inflate the
    reported score by an amount that cannot be recovered afterwards.
    """
    w = class_weights(ytr)[ytr]

    def objective(trial):
        model = xgb.XGBClassifier(**_xgb_params(trial, seed))
        model.fit(Xtr, ytr, sample_weight=w, verbose=False)
        return evaluate(yva, model.predict(Xva))["f1_macro"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def train_xgboost(Xtr, ytr, Xva, yva, Xte, yte, params, seed=C.PRIMARY_SEED):
    p = dict(params)
    p.update({"random_state": seed, "eval_metric": "mlogloss",
              "n_jobs": -1, "tree_method": "hist"})
    w = class_weights(ytr)[ytr]

    with timer() as t:
        model = xgb.XGBClassifier(**p)
        model.fit(Xtr, ytr, sample_weight=w, verbose=False)
    train_s = t()

    res = {name: evaluate(y, model.predict(X), model.predict_proba(X))
           for name, X, y in (("val", Xva, yva), ("test", Xte, yte))}
    n_params = int(sum(len(d) for d in model.get_booster().get_dump()))
    return model, res, train_s, n_params


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=C.N_TRIALS_XGB)
    ap.add_argument("--quick", action="store_true",
                    help="5 tuning trials and 1 seed - for smoke-testing only")
    args = ap.parse_args()

    n_trials = 5 if args.quick else args.trials
    seeds = [C.PRIMARY_SEED] if args.quick else C.MODEL_SEEDS

    print("=" * 70)
    print("STEP 5 - BASELINE MODELS")
    print("=" * 70)

    Xtr, ytr, Xva, yva, Xte, yte, meta = load_prepared()
    print(f"\ntrain {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    summary = {}

    # ---- Logistic Regression --------------------------------------------
    print("\n" + "-" * 70)
    print("Logistic Regression (performance floor)")
    print("-" * 70)
    runs = []
    for seed in seeds:
        model, scaler, res, secs, npar = train_logistic(
            Xtr, ytr, Xva, yva, Xte, yte, seed=seed)
        runs.append({**res["test"], "train_seconds": secs, "n_params": npar})
        log_experiment({"model": "LogisticRegression", "seed": seed,
                        "val_f1_macro": res["val"]["f1_macro"],
                        "test_f1_macro": res["test"]["f1_macro"],
                        "train_seconds": round(secs, 3), "n_params": npar,
                        "params": "class_weight=balanced"})
        print(f"  seed {seed:>5}  val F1={res['val']['f1_macro']:.4f}  "
              f"test F1={res['test']['f1_macro']:.4f}")
    summary["LogisticRegression"] = aggregate_seeds(runs)
    joblib.dump({"model": model, "scaler": scaler}, C.MODELS_DIR / "logistic_model.joblib")
    print_report(yte, model.predict(scaler.transform(Xte)),
                 "Logistic Regression - test set (last seed)")

    # ---- XGBoost ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"XGBoost (tuning: {n_trials} Optuna trials on validation)")
    print("-" * 70)
    best_params, best_val = tune_xgboost(Xtr, ytr, Xva, yva,
                                         n_trials=n_trials, seed=C.PRIMARY_SEED)
    print(f"  best validation F1-macro = {best_val:.4f}")
    print(f"  best params: {json.dumps(best_params, indent=2)}")
    with open(C.MODELS_DIR / "xgboost_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    runs, best_model, best_seed_f1 = [], None, -1.0
    for seed in seeds:
        model, res, secs, npar = train_xgboost(
            Xtr, ytr, Xva, yva, Xte, yte, best_params, seed=seed)
        runs.append({**res["test"], "train_seconds": secs, "n_params": npar})
        log_experiment({"model": "XGBoost", "seed": seed,
                        "val_f1_macro": res["val"]["f1_macro"],
                        "test_f1_macro": res["test"]["f1_macro"],
                        "train_seconds": round(secs, 3), "n_params": npar,
                        "params": json.dumps(best_params)})
        print(f"  seed {seed:>5}  val F1={res['val']['f1_macro']:.4f}  "
              f"test F1={res['test']['f1_macro']:.4f}")
        # keep the model selected on VALIDATION, never on test
        if res["val"]["f1_macro"] > best_seed_f1:
            best_seed_f1, best_model = res["val"]["f1_macro"], model

    summary["XGBoost"] = aggregate_seeds(runs)
    best_model.save_model(C.MODELS_DIR / "xgboost_model.json")
    print_report(yte, best_model.predict(Xte), "XGBoost - test set (best-val seed)")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY (test set, mean +/- std over seeds)")
    print("=" * 70)
    for name, s in summary.items():
        print(f"  {name:<22} F1-macro = {s['f1_macro_mean']:.4f} "
              f"+/- {s['f1_macro_std']:.4f}   "
              f"Healthy F1 = {s['f1_healthy_mean']:.4f}")

    with open(C.RESULTS_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved {(C.RESULTS_DIR / 'baseline_summary.json').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
