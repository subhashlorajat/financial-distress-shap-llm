"""
Hand-off validation - run this before telling Member 3 the work is ready.

Simulates what Member 3 will actually do: load artefacts, build features,
predict with every model, and run a real SHAP explanation. If this passes,
their SHAP and LLM narrative work has a stable foundation.

Run:  python src/validate_handoff.py

H9DLGA Group Project - Member 2 (Model Engineer).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import config as C

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    status = PASS if condition else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    symbol = {PASS: "[ok]", FAIL: "[FAIL]", WARN: "[warn]"}[status]
    print(f"  {symbol:7} {name}" + (f"  -- {detail}" if detail else ""))
    return condition


def main() -> None:
    print("=" * 70)
    print("HAND-OFF VALIDATION (Member 2 -> Member 3)")
    print("=" * 70)

    # ---- 1. artefacts exist ---------------------------------------------
    print("\n1. Required artefacts")
    required = {
        "splits.json": C.SPLITS_JSON,
        "preprocessor.joblib": C.PREPROCESSOR_PKL,
        "label_encoder.joblib": C.LABEL_ENCODER_PKL,
        "feature_columns.json": C.FEATURE_COLS_JSON,
        "background_sample.csv": C.BACKGROUND_CSV,
        "model_comparison.csv": C.COMPARISON_CSV,
    }
    for name, path in required.items():
        check(name, path.exists(), str(path.relative_to(C.ROOT)))

    # ---- 2. split integrity ---------------------------------------------
    print("\n2. Split integrity")
    with open(C.SPLITS_JSON) as f:
        splits = json.load(f)
    tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    check("no train/val company overlap", not (tr & va))
    check("no train/test company overlap", not (tr & te))
    check("no val/test company overlap", not (va & te))
    check("split seed recorded", "seed" in splits, f"seed={splits.get('seed')}")

    # ---- 3. prediction interface ----------------------------------------
    print("\n3. Prediction interface")
    from predict import (build_features, class_names, feature_columns,
                         load_background, load_model, load_test_set, predict)

    classes = class_names()
    cols = feature_columns()
    check("class order fixed", classes == C.CLASS_NAMES, str(classes))
    check("no leak column in features",
          not any(c in cols for c in C.LEAK_COLS),
          f"{len(cols)} feature columns")

    raw, X, y_true = load_test_set()
    check("test set loads", len(raw) > 0, f"{len(raw)} rows")
    check("matrix column order matches contract", list(X.columns) == cols)
    check("no NaN in matrix", not X.isna().any().any())
    check("all values finite", bool(np.isfinite(X.values).all()))

    bg = load_background()
    check("SHAP background sample usable",
          list(bg.columns) == cols and len(bg) > 0, f"{len(bg)} rows")

    # ---- 4. every model round-trips -------------------------------------
    print("\n4. Model loading and prediction")
    available = []
    for name in ("logistic", "xgboost", "mlp", "tabtransformer"):
        try:
            model = model = load_model(name)
            labels, proba, Xf = predict(model, raw.head(20))
            ok = (len(labels) == 20 and proba.shape == (20, 3)
                  and np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))
            check(f"{name} predicts correctly", ok,
                  f"probabilities sum to 1, shape {proba.shape}")
            available.append(name)
        except FileNotFoundError:
            check(f"{name} checkpoint", False, "not trained", warn_only=True)
        except Exception as exc:  # noqa: BLE001
            check(f"{name} predicts correctly", False, f"{type(exc).__name__}: {exc}")

    # ---- 5. determinism --------------------------------------------------
    print("\n5. Determinism")
    if available:
        model = load_model(available[0])
        _, p1, _ = predict(model, raw.head(50))
        _, p2, _ = predict(model, raw.head(50))
        check("repeated prediction is identical", np.allclose(p1, p2),
              f"model={available[0]}")

        # single row must match the same row inside a batch
        _, p_batch, _ = predict(model, raw.head(10))
        _, p_single, _ = predict(model, raw.iloc[[3]])
        check("single-row == batch-row prediction",
              np.allclose(p_batch[3], p_single[0], atol=1e-4),
              "row order preserved")

    # ---- 6. live SHAP smoke test ----------------------------------------
    print("\n6. SHAP compatibility")
    try:
        import shap
        model = load_model("xgboost")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.head(20))
        sv = np.asarray(sv)
        check("TreeSHAP runs on XGBoost", sv.size > 0, f"shape {sv.shape}")
        check("SHAP feature count matches matrix",
              sv.shape[1] == len(cols) or sv.shape[-1] == len(cols))
    except ImportError:
        check("shap installed", False, "pip install shap", warn_only=True)
    except Exception as exc:  # noqa: BLE001
        check("TreeSHAP runs on XGBoost", False, f"{type(exc).__name__}: {exc}")

    # ---- verdict ---------------------------------------------------------
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    n_pass = sum(1 for s, _, _ in results if s == PASS)

    print("\n" + "=" * 70)
    print(f"{n_pass} passed | {n_warn} warnings | {n_fail} failures")
    if n_fail == 0:
        print("HAND-OFF READY - Member 3 can begin.")
    else:
        print("HAND-OFF BLOCKED - resolve failures above.")
    print("=" * 70)

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
