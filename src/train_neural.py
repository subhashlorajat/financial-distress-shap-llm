"""
Steps 5-6 - Neural models: MLP baseline and TabTransformer.

Both models share nn_trainer.train_model, so the RQ1 comparison isolates the
architecture rather than the training budget.

Input format differs by design, and this is the point of the experiment:
  MLP             one-hot matrix (23 columns) - the same input XGBoost sees
  TabTransformer  integer-encoded categoricals through embedding tables and
                  self-attention, continuous features concatenated afterwards

Run:  python src/train_neural.py [--trials 25] [--quick]

H9DLGA Group Project - Member 2 (Model Engineer).
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import optuna
import torch

import config as C
from data_prep import load_prepared
from evaluation import (aggregate_seeds, evaluate, log_experiment,
                        print_report, timer)
from features import CATEGORICALS, NUMERIC
from nn_models import MLP, TabTransformer, count_parameters, set_all_seeds
from nn_trainer import (build_categorical_encoder, cardinalities,
                        fit_continuous_scaler, make_onehot_tensors,
                        make_tensors, predict_proba, train_model)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------
def _mlp_packs(Xtr, ytr, Xva, yva, Xte, yte):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)              # TRAIN only
    packs = []
    for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte)):
        xc, xk = make_onehot_tensors(scaler.transform(X))
        packs.append((xc, xk, torch.tensor(y, dtype=torch.long)))
    return packs, scaler


def tune_mlp(packs, n_trials, seed):
    tr, va, _ = packs
    n_features = tr[0].shape[1]

    def objective(trial):
        set_all_seeds(seed)
        hidden = trial.suggest_categorical(
            "hidden", ["256-128", "128-64", "256-128-64", "64-32"])
        model = MLP(n_features,
                    hidden_dims=tuple(int(h) for h in hidden.split("-")),
                    dropout=trial.suggest_float("dropout", 0.1, 0.5))
        _, _, best = train_model(
            model, tr, va, device=DEVICE,
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            max_epochs=60, patience=10)
        return best

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def train_mlp_final(packs, params, seed):
    set_all_seeds(seed)
    tr, va, te = packs
    model = MLP(tr[0].shape[1],
                hidden_dims=tuple(int(h) for h in params["hidden"].split("-")),
                dropout=params["dropout"])
    with timer() as t:
        model, history, best_val = train_model(
            model, tr, va, device=DEVICE, lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"])
    secs = t()

    res = {}
    for name, pack in (("val", va), ("test", te)):
        proba = predict_proba(model, pack[0], pack[1], device=DEVICE)
        res[name] = evaluate(pack[2].numpy(), proba.argmax(1), proba)
    return model, res, history, secs, count_parameters(model)


# --------------------------------------------------------------------------
# TabTransformer
# --------------------------------------------------------------------------
def _tt_packs(meta, ytr, yva, yte):
    train_df, val_df, test_df = meta["train_df"], meta["val_df"], meta["test_df"]
    scaler = fit_continuous_scaler(train_df)        # TRAIN only
    enc = build_categorical_encoder(train_df)       # TRAIN only
    packs = []
    for df, y in ((train_df, ytr), (val_df, yva), (test_df, yte)):
        xc, xk = make_tensors(df, scaler, enc)
        packs.append((xc, xk, torch.tensor(y, dtype=torch.long)))
    return packs, scaler, enc


def tune_tabtransformer(packs, cards, n_trials, seed):
    tr, va, _ = packs
    n_cont = tr[0].shape[1]

    def objective(trial):
        set_all_seeds(seed)
        embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64])
        valid_heads = [h for h in (2, 4, 8) if embed_dim % h == 0]
        model = TabTransformer(
            cards, n_cont, embed_dim=embed_dim,
            n_heads=trial.suggest_categorical("n_heads", valid_heads),
            n_blocks=trial.suggest_int("n_blocks", 2, 6),
            attn_dropout=trial.suggest_float("attn_dropout", 0.0, 0.3),
            mlp_dropout=trial.suggest_float("mlp_dropout", 0.1, 0.5))
        _, _, best = train_model(
            model, tr, va, device=DEVICE,
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            max_epochs=60, patience=10)
        return best

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def train_tt_final(packs, cards, params, seed):
    set_all_seeds(seed)
    tr, va, te = packs
    model = TabTransformer(
        cards, tr[0].shape[1], embed_dim=params["embed_dim"],
        n_heads=params["n_heads"], n_blocks=params["n_blocks"],
        attn_dropout=params["attn_dropout"], mlp_dropout=params["mlp_dropout"])
    with timer() as t:
        model, history, best_val = train_model(
            model, tr, va, device=DEVICE, lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"])
    secs = t()

    res = {}
    for name, pack in (("val", va), ("test", te)):
        proba = predict_proba(model, pack[0], pack[1], device=DEVICE)
        res[name] = evaluate(pack[2].numpy(), proba.argmax(1), proba)
    return model, res, history, secs, count_parameters(model)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=C.N_TRIALS_MLP)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_trials = 3 if args.quick else args.trials
    seeds = [C.PRIMARY_SEED] if args.quick else C.MODEL_SEEDS

    print("=" * 70)
    print(f"STEPS 5-6 - NEURAL MODELS  (device: {DEVICE})")
    print("=" * 70)

    Xtr, ytr, Xva, yva, Xte, yte, meta = load_prepared()
    summary = {}

    # ---- MLP -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"MLP baseline (tuning: {n_trials} trials on validation)")
    print("-" * 70)
    packs, mlp_scaler = _mlp_packs(Xtr, ytr, Xva, yva, Xte, yte)
    best_params, best_val = tune_mlp(packs, n_trials, C.PRIMARY_SEED)
    print(f"  best validation F1-macro = {best_val:.4f}")
    print(f"  best params: {json.dumps(best_params)}")

    runs, best_model, best_hist, best_sel = [], None, None, -1.0
    for seed in seeds:
        model, res, hist, secs, npar = train_mlp_final(packs, best_params, seed)
        runs.append({**res["test"], "train_seconds": secs, "n_params": npar})
        log_experiment({"model": "MLP", "seed": seed,
                        "val_f1_macro": res["val"]["f1_macro"],
                        "test_f1_macro": res["test"]["f1_macro"],
                        "train_seconds": round(secs, 2), "n_params": npar,
                        "params": json.dumps(best_params)})
        print(f"  seed {seed:>5}  val F1={res['val']['f1_macro']:.4f}  "
              f"test F1={res['test']['f1_macro']:.4f}  ({secs:.1f}s)")
        if res["val"]["f1_macro"] > best_sel:      # selected on VALIDATION
            best_sel, best_model, best_hist = res["val"]["f1_macro"], model, hist

    summary["MLP"] = aggregate_seeds(runs)
    torch.save({"state_dict": best_model.state_dict(), "params": best_params,
                "n_features": packs[0][0].shape[1]}, C.MODELS_DIR / "mlp_model.pt")
    joblib.dump(mlp_scaler, C.MODELS_DIR / "mlp_scaler.joblib")
    best_hist.to_csv(C.RESULTS_DIR / "mlp_training_history.csv", index=False)
    with open(C.MODELS_DIR / "mlp_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    proba = predict_proba(best_model, packs[2][0], packs[2][1], device=DEVICE)
    print_report(yte, proba.argmax(1), "MLP - test set (best-val seed)")

    # ---- TabTransformer --------------------------------------------------
    print("\n" + "-" * 70)
    print(f"TabTransformer (tuning: {n_trials} trials on validation)")
    print("-" * 70)
    tt_packs, tt_scaler, tt_enc = _tt_packs(meta, ytr, yva, yte)
    cards = cardinalities(tt_enc)
    print(f"  categorical cardinalities {dict(zip(CATEGORICALS, cards))}")
    print(f"  continuous features: {len(NUMERIC)}")

    best_params, best_val = tune_tabtransformer(tt_packs, cards, n_trials,
                                                C.PRIMARY_SEED)
    print(f"  best validation F1-macro = {best_val:.4f}")
    print(f"  best params: {json.dumps(best_params)}")

    runs, best_model, best_hist, best_sel = [], None, None, -1.0
    for seed in seeds:
        model, res, hist, secs, npar = train_tt_final(tt_packs, cards,
                                                      best_params, seed)
        runs.append({**res["test"], "train_seconds": secs, "n_params": npar})
        log_experiment({"model": "TabTransformer", "seed": seed,
                        "val_f1_macro": res["val"]["f1_macro"],
                        "test_f1_macro": res["test"]["f1_macro"],
                        "train_seconds": round(secs, 2), "n_params": npar,
                        "params": json.dumps(best_params)})
        print(f"  seed {seed:>5}  val F1={res['val']['f1_macro']:.4f}  "
              f"test F1={res['test']['f1_macro']:.4f}  ({secs:.1f}s)")
        if res["val"]["f1_macro"] > best_sel:
            best_sel, best_model, best_hist = res["val"]["f1_macro"], model, hist

    summary["TabTransformer"] = aggregate_seeds(runs)
    torch.save({"state_dict": best_model.state_dict(), "params": best_params,
                "cardinalities": cards, "n_continuous": tt_packs[0][0].shape[1]},
               C.MODELS_DIR / "tabtransformer_model.pt")
    joblib.dump({"scaler": tt_scaler, "encoder": tt_enc},
                C.MODELS_DIR / "tabtransformer_preproc.joblib")
    best_hist.to_csv(C.RESULTS_DIR / "tabtransformer_training_history.csv",
                     index=False)
    with open(C.MODELS_DIR / "tabtransformer_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    proba = predict_proba(best_model, tt_packs[2][0], tt_packs[2][1], device=DEVICE)
    print_report(yte, proba.argmax(1), "TabTransformer - test set (best-val seed)")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("NEURAL SUMMARY (test set, mean +/- std over seeds)")
    print("=" * 70)
    for name, s in summary.items():
        print(f"  {name:<22} F1-macro = {s['f1_macro_mean']:.4f} "
              f"+/- {s['f1_macro_std']:.4f}   "
              f"Healthy F1 = {s['f1_healthy_mean']:.4f}")

    with open(C.RESULTS_DIR / "neural_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved {(C.RESULTS_DIR / 'neural_summary.json').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
