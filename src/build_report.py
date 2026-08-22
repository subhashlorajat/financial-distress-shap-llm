
# Model comparison report and paper figures.
# Member 2 (Model Engineer)

from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

import config as C
from predict import (build_features, class_names, feature_columns, load_model,
                     load_test_set)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

MODEL_ORDER = ["LogisticRegression", "XGBoost", "MLP", "TabTransformer"]
MODEL_KEYS = {"LogisticRegression": "logistic", "XGBoost": "xgboost",
              "MLP": "mlp", "TabTransformer": "tabtransformer"}
PALETTE = {"LogisticRegression": "#9aa0a6", "XGBoost": "#2a9d8f",
           "MLP": "#e9c46a", "TabTransformer": "#e76f51"}


def load_summaries() -> dict:
    merged = {}
    for path in (C.RESULTS_DIR / "baseline_summary.json",
                 C.RESULTS_DIR / "neural_summary.json"):
        if path.exists():
            with open(path) as f:
                merged.update(json.load(f))
    if not merged:
        raise FileNotFoundError(
            "No summaries found. Run train_baselines.py and train_neural.py first.")
    return merged


def build_comparison_table(summaries: dict) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        if name not in summaries:
            continue
        s = summaries[name]
        rows.append({
            "model": name,
            "f1_macro_mean": s["f1_macro_mean"],
            "f1_macro_std": s["f1_macro_std"],
            "auc_roc_macro_mean": s.get("auc_roc_macro_mean", np.nan),
            "auc_roc_macro_std": s.get("auc_roc_macro_std", np.nan),
            "accuracy_mean": s["accuracy_mean"],
            "f1_at_risk_mean": s["f1_at_risk_mean"],
            "f1_distressed_mean": s["f1_distressed_mean"],
            "f1_healthy_mean": s["f1_healthy_mean"],
            "recall_healthy_mean": s["recall_healthy_mean"],
            "precision_healthy_mean": s["precision_healthy_mean"],
            "train_seconds_mean": s["train_seconds_mean"],
            "n_params_mean": s["n_params_mean"],
        })
    return pd.DataFrame(rows)


def paper_table(df: pd.DataFrame) -> pd.DataFrame:
    """Trimmed table formatted for direct paste into the IEEE paper."""
    out = pd.DataFrame({
        "Model": df.model,
        "F1-macro": [f"{m:.3f} ± {s:.3f}" for m, s in
                     zip(df.f1_macro_mean, df.f1_macro_std)],
        "AUC-ROC": [f"{m:.3f}" if not np.isnan(m) else "-"
                    for m in df.auc_roc_macro_mean],
        "F1 At-Risk": df.f1_at_risk_mean.round(3),
        "F1 Distressed": df.f1_distressed_mean.round(3),
        "F1 Healthy": df.f1_healthy_mean.round(3),
        "Train (s)": df.train_seconds_mean.round(1),
        "Params": df.n_params_mean.astype(int),
    })
    return out


# --------------------------------------------------------------------------
def fig_f1_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar(df.model, df.f1_macro_mean, yerr=df.f1_macro_std, capsize=4,
           color=[PALETTE[m] for m in df.model], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("F1-macro (test)")
    ax.set_title("Overall performance (mean ± std over seeds)")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=15)
    for i, (m, s) in enumerate(zip(df.f1_macro_mean, df.f1_macro_std)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    classes = class_names()
    width = 0.8 / len(df)
    x = np.arange(len(classes))
    for i, row in df.iterrows():
        vals = [row.f1_at_risk_mean, row.f1_distressed_mean, row.f1_healthy_mean]
        ax.bar(x + i * width, vals, width, label=row.model,
               color=PALETTE[row.model], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 (test)")
    ax.set_title("Per-class performance")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "f1_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_roc(probas: dict, y_true: np.ndarray) -> None:
    classes = class_names()
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, axes = plt.subplots(1, len(classes), figsize=(13, 4))

    for k, cls in enumerate(classes):
        ax = axes[k]
        for name, proba in probas.items():
            fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, k])
            ax.plot(fpr, tpr, label=f"{name} ({auc(fpr, tpr):.3f})",
                    color=PALETTE[name], linewidth=1.4)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{cls} (one-vs-rest)")
        ax.set_xlabel("False positive rate")
        if k == 0:
            ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(preds: dict, y_true: np.ndarray) -> None:
    classes = class_names()
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6))
    axes = np.atleast_1d(axes)

    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=30, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        if ax is axes[0]:
            ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(C.FIGURES_DIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def fig_training_curves() -> None:
    files = {"MLP": C.RESULTS_DIR / "mlp_training_history.csv",
             "TabTransformer": C.RESULTS_DIR / "tabtransformer_training_history.csv"}
    available = {k: v for k, v in files.items() if v.exists()}
    if not available:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, path in available.items():
        h = pd.read_csv(path)
        axes[0].plot(h.epoch, h.train_loss, label=f"{name} train",
                     color=PALETTE[name], linewidth=1.2)
        axes[0].plot(h.epoch, h.val_loss, label=f"{name} val",
                     color=PALETTE[name], linewidth=1.2, linestyle="--")
        axes[1].plot(h.epoch, h.val_f1_macro, label=name,
                     color=PALETTE[name], linewidth=1.4)
        best = h.val_f1_macro.idxmax()
        axes[1].scatter(h.epoch[best], h.val_f1_macro[best],
                        color=PALETTE[name], s=30, zorder=5)

    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training / validation loss")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set(xlabel="Epoch", ylabel="F1-macro", title="Validation F1-macro (dot = best epoch)")
    axes[1].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_feature_importance() -> None:
    """XGBoost gain importance.

    Also the circularity audit: labels derive from Altman Z', so if one feature
    dominated (>60% of total gain) it would suggest the model had found a proxy
    for the label rather than learning the task.
    """
    try:
        model = load_model("xgboost")
    except FileNotFoundError:
        return

    imp = pd.Series(model.feature_importances_,
                    index=feature_columns()).sort_values(ascending=False)
    top = imp.head(15)[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#2a9d8f", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Gain importance")
    ax.set_title(f"XGBoost feature importance (top feature: {imp.iloc[0]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    imp.to_csv(C.RESULTS_DIR / "feature_importance.csv", header=["gain"])
    return imp


# ==========================
def main() -> None:
    print("=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)

    summaries = load_summaries()
    df = build_comparison_table(summaries)
    df.to_csv(C.COMPARISON_CSV, index=False)

    tbl = paper_table(df)
    tbl.to_csv(C.RESULTS_DIR / "model_comparison_paper.csv", index=False)
    print("\n" + tbl.to_string(index=False))

    # ---- test-set predictions from every trained model -------------------
    raw, X, y_true = load_test_set()
    probas, preds = {}, {}
    for name in MODEL_ORDER:
        if name not in summaries:
            continue
        try:
            model = load_model(MODEL_KEYS[name])
            p = model.predict_proba(X)
            probas[name] = p
            preds[name] = p.argmax(axis=1)
        except FileNotFoundError:
            print(f"  (skipping {name} - checkpoint missing)")

    # per-row predictions for Member 3's error analysis
    out = raw[["company_id", "year", "label"]].copy()
    for name, p in preds.items():
        out[f"pred_{name}"] = np.array(class_names())[p]
        out[f"conf_{name}"] = probas[name].max(axis=1).round(4)
    out.to_csv(C.PREDICTIONS_CSV, index=False)

    # ---- figures ---------------------------------------------------------
    fig_f1_comparison(df)
    fig_roc(probas, y_true)
    fig_confusion(preds, y_true)
    fig_training_curves()
    imp = fig_feature_importance()

    print("\nFigures written to results/figures/:")
    for p in sorted(C.FIGURES_DIR.glob("*.png")):
        print(f"  {p.name}")

    # ---- headline --------------------------------------------------------
    best = df.loc[df.f1_macro_mean.idxmax()]
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best.model}   F1-macro = {best.f1_macro_mean:.4f} "
          f"± {best.f1_macro_std:.4f}")
    print("=" * 70)

    if imp is not None:
        share = imp.iloc[0] * 100
        verdict = "PASS - no single feature dominates" if share < 60 else \
                  "WARNING - possible label proxy, investigate"
        print(f"Circularity audit: top feature '{imp.index[0]}' = {share:.1f}% "
              f"of gain -> {verdict}")

    # RQ1 verdict, stated honestly either way
    if {"XGBoost", "TabTransformer"} <= set(df.model):
        xgb_row = df[df.model == "XGBoost"].iloc[0]
        tt_row = df[df.model == "TabTransformer"].iloc[0]
        gap = tt_row.f1_macro_mean - xgb_row.f1_macro_mean
        pooled = np.sqrt(xgb_row.f1_macro_std**2 + tt_row.f1_macro_std**2)
        print(f"\nRQ1  TabTransformer - XGBoost = {gap:+.4f} "
              f"(pooled std {pooled:.4f})")
        if abs(gap) < max(pooled, 0.01):
            print("     -> difference is within noise; no evidence either model wins")
        elif gap > 0:
            print("     -> TabTransformer outperforms gradient boosting")
        else:
            print("     -> gradient boosting outperforms TabTransformer")


if __name__ == "__main__":
    main()


# Produces the headline artefacts:
    # results/model_comparison.csv          full metric table
    # results/model_comparison_paper.csv    trimmed, publication-ready
    # results/test_predictions.csv          per-row predictions for error analysis
    # results/figures/f1_comparison.png     F1-macro with std error bars
    # results/figures/roc_curves.png        one-vs-rest ROC overlay
    # results/figures/confusion_matrices.png
    # results/figures/training_curves.png
    # results/figures/feature_importance.png

