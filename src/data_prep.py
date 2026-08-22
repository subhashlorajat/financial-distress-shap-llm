# Step 4 - Data preparation and splitting.

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config as C
from features import (CATEGORICALS, NUMERIC, add_numeric_features,
                      apply_preprocessor, fit_preprocessor, load_raw)
from splitting import (apply_splits, assert_no_overlap, make_company_splits,
                       save_splits)


# create model input matrix
def build_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert categorical values and return model-ready data.

    columns keeps the feature order consistent between train, validation,
    and test sets.
    """
    X = pd.get_dummies(df[NUMERIC + CATEGORICALS], columns=CATEGORICALS)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.astype(np.float32)


def _class_table(name: str, df: pd.DataFrame) -> dict:
    pct = (df.label.value_counts(normalize=True) * 100).round(1)
    return {"split": name, "rows": len(df), "companies": df.company_id.nunique(),
            **{c: pct.get(c, 0.0) for c in C.CLASS_NAMES}}


def main() -> None:
    print("=" * 70)
    print("STEP 4 - DATA PREPARATION AND SPLITTING")
    print("=" * 70)

    # load data and create extra features
    df = load_raw(C.RAW_CSV)
    df = add_numeric_features(df)
    print(f"\nLoaded {len(df):,} rows / {df.company_id.nunique():,} companies "
          f"(FY{df.year.min()}-{df.year.max()}, {C.DROP_YEARS} dropped)")

    # keep companies in only one split
    splits = make_company_splits(df, seed=C.PRIMARY_SEED,
                                 val_size=C.VAL_SIZE, test_size=C.TEST_SIZE)
    assert_no_overlap(splits)
    save_splits(splits, C.SPLITS_JSON, C.PRIMARY_SEED)
    print(f"Saved {C.SPLITS_JSON.relative_to(C.ROOT)} (seed={C.PRIMARY_SEED})")

    train_df, val_df, test_df = apply_splits(df, splits)

    # fit preprocessing using training data only
    prep = fit_preprocessor(train_df)
    train_df = apply_preprocessor(train_df, prep)
    val_df = apply_preprocessor(val_df, prep)
    test_df = apply_preprocessor(test_df, prep)

    # build feature columns used by models
    X_train = build_matrix(train_df)
    feature_columns = X_train.columns.tolist()
    X_val = build_matrix(val_df, feature_columns)
    X_test = build_matrix(test_df, feature_columns)

    le = LabelEncoder().fit(C.CLASS_NAMES)

    # basic checks before saving outputs
    assert "z_score" not in feature_columns, "z_score leaked into features"
    assert not any(c in feature_columns for c in C.LEAK_COLS), "leak column present"
    for name, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        assert not X.isna().any().any(), f"NaNs in {name}"
        assert np.isfinite(X.values).all(), f"non-finite values in {name}"
    assert list(X_val.columns) == feature_columns == list(X_test.columns)

    for c in CATEGORICALS:
        unseen = (set(val_df[c]) | set(test_df[c])) - set(train_df[c])
        assert not unseen, f"unseen level in {c}: {unseen}"

    print("\nIntegrity gates: PASSED")
    print("  no leak columns | no NaN/inf | column order pinned | no unseen levels")

    # split statistics
    table = pd.DataFrame([_class_table("train", train_df),
                          _class_table("val", val_df),
                          _class_table("test", test_df)])
    print("\nSplit summary")
    print(table.to_string(index=False))
    print(f"\nFeature matrix: {len(NUMERIC)} numeric + {len(CATEGORICALS)} "
          f"categorical -> {len(feature_columns)} columns after one-hot")

    # save files for training and explainability
    joblib.dump(prep, C.PREPROCESSOR_PKL)
    joblib.dump(le, C.LABEL_ENCODER_PKL)

    with open(C.FEATURE_COLS_JSON, "w") as f:
        json.dump(feature_columns, f, indent=2)

    # sample data used by SHAP
    bg = X_train.sample(n=min(100, len(X_train)), random_state=C.PRIMARY_SEED)
    bg.to_csv(C.BACKGROUND_CSV, index=False)

    print("\nArtefacts written for Member 3:")
    for p in (C.PREPROCESSOR_PKL, C.LABEL_ENCODER_PKL,
              C.FEATURE_COLS_JSON, C.BACKGROUND_CSV, C.SPLITS_JSON):
        print(f"  {p.relative_to(C.ROOT)}")


def load_prepared(seed: int | None = None):
    """Load prepared data for training scripts.

    Returns processed datasets with metadata required for evaluation.
    """
    df = add_numeric_features(load_raw(C.RAW_CSV))

    with open(C.SPLITS_JSON) as f:
        payload = json.load(f)

    splits = {k: payload[k] for k in ("train", "val", "test")}

    train_df, val_df, test_df = apply_splits(df, splits)

    prep = joblib.load(C.PREPROCESSOR_PKL)
    train_df = apply_preprocessor(train_df, prep)
    val_df = apply_preprocessor(val_df, prep)
    test_df = apply_preprocessor(test_df, prep)

    with open(C.FEATURE_COLS_JSON) as f:
        cols = json.load(f)

    le = joblib.load(C.LABEL_ENCODER_PKL)

    X_train = build_matrix(train_df, cols)
    X_val = build_matrix(val_df, cols)
    X_test = build_matrix(test_df, cols)

    y_train = le.transform(train_df.label)
    y_val = le.transform(val_df.label)
    y_test = le.transform(test_df.label)

    meta = {"label_encoder": le, "feature_columns": cols,
            "train_df": train_df, "val_df": val_df, "test_df": test_df,
            "preprocessor": prep}

    return X_train, y_train, X_val, y_val, X_test, y_test, meta


if __name__ == "__main__":
    main()