"""
Shared evaluation utilities.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config as C


@contextmanager
def timer():
    """Simple wall-clock timer."""
    start = time.perf_counter()
    elapsed = {}
    yield lambda: elapsed.get("s", time.perf_counter() - start)
    elapsed["s"] = time.perf_counter() - start


def evaluate(y_true, y_pred, y_proba=None, class_names=None) -> dict:
    """Calculate evaluation metrics for a model run.

    y_proba is used when AUC-ROC calculation is required.
    """
    class_names = class_names or C.CLASS_NAMES

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None,
                            labels=range(len(class_names)), zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None,
                                 labels=range(len(class_names)), zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None,
                                     labels=range(len(class_names)), zero_division=0)

    for i, name in enumerate(class_names):
        key = name.lower().replace("-", "_")
        out[f"f1_{key}"] = per_class_f1[i]
        out[f"recall_{key}"] = per_class_rec[i]
        out[f"precision_{key}"] = per_class_prec[i]

    if y_proba is not None:
        try:
            out["auc_roc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            # missing class in split
            out["auc_roc_macro"] = np.nan

    return out


def print_report(y_true, y_pred, title="", class_names=None) -> None:
    class_names = class_names or C.CLASS_NAMES

    if title:
        print(f"\n{title}")
        print("-" * len(title))

    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    print("Confusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())


def aggregate_seeds(rows: list[dict]) -> dict:
    """Combine multiple seed results using mean and standard deviation."""
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number])

    out = {}
    for col in numeric.columns:
        out[f"{col}_mean"] = numeric[col].mean()
        out[f"{col}_std"] = numeric[col].std(ddof=1) if len(numeric) > 1 else 0.0

    return out


def log_experiment(record: dict, path=None) -> None:
    """Append experiment details to the csv log."""
    path = path or C.EXPERIMENT_LOG_CSV

    record = {"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
              **record}

    df = pd.DataFrame([record])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def class_weights(y) -> np.ndarray:
    """Generate balanced weights for each class."""
    counts = np.bincount(y, minlength=len(C.CLASS_NAMES)).astype(float)
    counts[counts == 0] = 1.0
    return len(y) / (len(C.CLASS_NAMES) * counts)
