"""
Prediction interface - the single entry point for Member 3 (SHAP + LLM).

Why go through this module rather than loading a model directly: SHAP explains
whatever matrix it is handed, so the matrix must be byte-identical to the one
the model was trained on. That means the same preprocessing constants, the same
one-hot expansion and - critically - the same column order. This module pins
all three.

Typical use
-----------
    import sys; sys.path.insert(0, "src")
    from predict import load_model, predict, build_features, class_names

    model = load_model("xgboost")
    labels, proba, X = predict(model, raw_rows)     # X is what SHAP explains

    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

H9DLGA Group Project - Member 2 (Model Engineer).
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import config as C
from features import (CATEGORICALS, NUMERIC, add_numeric_features,
                      apply_preprocessor, load_raw)


# --------------------------------------------------------------------------
# Cached artefacts
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _artefacts():
    with open(C.FEATURE_COLS_JSON) as f:
        columns = json.load(f)
    return {"preprocessor": joblib.load(C.PREPROCESSOR_PKL),
            "label_encoder": joblib.load(C.LABEL_ENCODER_PKL),
            "columns": columns}


def class_names() -> list[str]:
    """Class order used by every model output. Index i == class_names()[i]."""
    return [str(c) for c in _artefacts()["label_encoder"].classes_]


def feature_columns() -> list[str]:
    """Exact column order of the model input matrix."""
    return list(_artefacts()["columns"])


def decode(indices) -> np.ndarray:
    """Class indices -> readable labels (plain Python strings)."""
    out = _artefacts()["label_encoder"].inverse_transform(np.asarray(indices))
    return np.array([str(v) for v in out], dtype=object)


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> model-ready matrix.

    ``raw_df`` must carry the columns of ``data/labelled_full.csv``. Engineered
    features, clipping, imputation and categorical binning are all replayed
    from the fitted preprocessor, so a single unseen row is treated exactly as
    a training row was.
    """
    art = _artefacts()
    d = apply_preprocessor(add_numeric_features(raw_df.copy()),
                           art["preprocessor"])
    X = pd.get_dummies(d[NUMERIC + CATEGORICALS], columns=CATEGORICALS)
    return X.reindex(columns=art["columns"], fill_value=0).astype(np.float32)


def load_test_set() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """The held-out test split: (raw_rows, feature_matrix, true_label_indices).

    Member 3 should explain predictions on this split so that the narrative
    evaluation runs on data no model ever saw.
    """
    with open(C.SPLITS_JSON) as f:
        test_companies = set(json.load(f)["test"])

    df = add_numeric_features(load_raw(C.RAW_CSV))
    raw = df[df.company_id.isin(test_companies)].reset_index(drop=True)
    X = build_features(raw)
    y = _artefacts()["label_encoder"].transform(raw.label)
    return raw, X, y


def load_background(n: int | None = None) -> pd.DataFrame:
    """Training-distribution sample for SHAP baselines (DeepSHAP/KernelSHAP)."""
    bg = pd.read_csv(C.BACKGROUND_CSV)
    return bg.head(n) if n else bg


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def load_model(name: str = "xgboost"):
    """Load a trained model by name.

    Returns an object exposing ``predict`` and ``predict_proba`` for every
    architecture, so downstream code does not branch on model type.
    Valid names: ``xgboost``, ``logistic``, ``mlp``, ``tabtransformer``.
    """
    name = name.lower()

    if name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(C.MODELS_DIR / "xgboost_model.json")
        return model

    if name == "logistic":
        bundle = joblib.load(C.MODELS_DIR / "logistic_model.joblib")
        return _SklearnWrapper(bundle["model"], bundle["scaler"])

    if name in ("mlp", "tabtransformer"):
        return _TorchWrapper(name)

    raise ValueError(f"unknown model '{name}'. Valid: xgboost, logistic, "
                     f"mlp, tabtransformer")


class _SklearnWrapper:
    """Applies the model's own scaler before delegating."""

    def __init__(self, model, scaler):
        self.model, self.scaler = model, scaler

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


class _TorchWrapper:
    """Gives the PyTorch models a scikit-learn style interface.

    The MLP consumes the one-hot matrix directly. TabTransformer needs integer
    categorical indices, so the one-hot block is inverted back to indices here -
    that keeps ``build_features`` as the single feature entry point and means
    SHAP still explains one consistent matrix across all four models.
    """

    def __init__(self, name):
        import torch
        from nn_models import MLP, TabTransformer

        self.name = name
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.columns = feature_columns()

        if name == "mlp":
            ckpt = torch.load(C.MODELS_DIR / "mlp_model.pt",
                              map_location=self.device, weights_only=False)
            p = ckpt["params"]
            self.model = MLP(ckpt["n_features"],
                             hidden_dims=tuple(int(h) for h in p["hidden"].split("-")),
                             dropout=p["dropout"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.scaler = joblib.load(C.MODELS_DIR / "mlp_scaler.joblib")
        else:
            ckpt = torch.load(C.MODELS_DIR / "tabtransformer_model.pt",
                              map_location=self.device, weights_only=False)
            p = ckpt["params"]
            self.model = TabTransformer(
                ckpt["cardinalities"], ckpt["n_continuous"],
                embed_dim=p["embed_dim"], n_heads=p["n_heads"],
                n_blocks=p["n_blocks"], attn_dropout=p["attn_dropout"],
                mlp_dropout=p["mlp_dropout"])
            self.model.load_state_dict(ckpt["state_dict"])
            pre = joblib.load(C.MODELS_DIR / "tabtransformer_preproc.joblib")
            self.scaler, self.encoder = pre["scaler"], pre["encoder"]

        self.model.to(self.device).eval()

    def _onehot_to_indices(self, X: pd.DataFrame):
        """Recover categorical indices from the one-hot block."""
        cols = []
        for cat in CATEGORICALS:
            prefix = f"{cat}_"
            members = [c for c in self.columns if c.startswith(prefix)]
            block = X[members].to_numpy()
            level_names = [c[len(prefix):] for c in members]
            picked = block.argmax(axis=1)
            mapping = self.encoder[cat]
            idx = np.array([mapping.get(level_names[p], 0) for p in picked])
            # a row with no active dummy is genuinely unseen -> OOV slot
            idx[block.sum(axis=1) == 0] = 0
            cols.append(idx)
        return np.stack(cols, axis=1).astype(np.int64)

    def predict_proba(self, X):
        torch = self.torch
        X = pd.DataFrame(X, columns=self.columns) if not isinstance(X, pd.DataFrame) else X

        if self.name == "mlp":
            xc = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            xk = torch.zeros((len(X), 0), dtype=torch.long)
        else:
            xc = torch.tensor(self.scaler.transform(X[NUMERIC]), dtype=torch.float32)
            xk = torch.tensor(self._onehot_to_indices(X), dtype=torch.long)

        out = []
        with torch.no_grad():
            for i in range(0, len(xc), 1024):
                logits = self.model(xc[i:i + 1024].to(self.device),
                                    xk[i:i + 1024].to(self.device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# --------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------
def predict(model, raw_df: pd.DataFrame):
    """Raw rows -> (labels, probabilities, feature_matrix).

    The third return value is the matrix SHAP must explain. Returning it
    alongside the prediction guarantees the explanation and the prediction refer
    to the same input.
    """
    X = build_features(raw_df)
    proba = model.predict_proba(X)
    labels = decode(proba.argmax(axis=1))
    return labels, proba, X


def explain_ready(model_name: str = "xgboost"):
    """Everything Member 3 needs in one call.

    Returns dict with: model, raw rows, feature matrix, true labels,
    predicted labels, probabilities, background sample, class names,
    feature column order.
    """
    model = load_model(model_name)
    raw, X, y_true = load_test_set()
    proba = model.predict_proba(X)
    return {"model": model, "model_name": model_name,
            "raw": raw, "X": X,
            "y_true": y_true, "y_true_labels": decode(y_true),
            "y_pred": proba.argmax(axis=1), "y_pred_labels": decode(proba.argmax(axis=1)),
            "proba": proba, "background": load_background(),
            "class_names": class_names(), "feature_columns": feature_columns()}


if __name__ == "__main__":
    # Smoke test - run after training to confirm the hand-off works.
    print("Prediction interface smoke test")
    print("-" * 50)
    print("classes:", class_names())
    print("features:", len(feature_columns()))

    raw, X, y = load_test_set()
    print(f"test set: {len(raw)} rows, matrix {X.shape}")

    for name in ("logistic", "xgboost", "mlp", "tabtransformer"):
        try:
            m = load_model(name)
            labels, proba, Xf = predict(m, raw.head(5))
            print(f"  {name:<16} OK  -> {list(labels)}")
        except FileNotFoundError:
            print(f"  {name:<16} not trained yet")
