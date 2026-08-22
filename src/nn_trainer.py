
# Member 2 (Model Engineer)
# Shared training loop for neural models.
# Used by MLP and TabTransformer training.

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config as C
from evaluation import class_weights, evaluate
from features import CATEGORICALS, NUMERIC


# tensor helpers
def build_categorical_encoder(train_df) -> dict:
    """Create categorical mappings using training data only.

    Index 0 is kept for unknown values.
    """
    enc = {}
    for col in CATEGORICALS:
        levels = sorted(train_df[col].astype(str).unique())
        enc[col] = {lvl: i + 1 for i, lvl in enumerate(levels)}
    return enc


def encode_categoricals(df, enc) -> np.ndarray:
    cols = [df[c].astype(str).map(lambda v, e=enc[c]: e.get(v, 0)).to_numpy()
            for c in CATEGORICALS]
    return np.stack(cols, axis=1).astype(np.int64)


def cardinalities(enc) -> list[int]:
    return [len(enc[c]) + 1 for c in CATEGORICALS]   # reserve index 0


def make_tensors(df, scaler, enc, continuous_cols=None):
    """Create tensors for TabTransformer input."""
    continuous_cols = continuous_cols or NUMERIC
    x_cont = torch.tensor(scaler.transform(df[continuous_cols]),
                          dtype=torch.float32)
    x_cat = torch.tensor(encode_categoricals(df, enc), dtype=torch.long)
    return x_cont, x_cat


def make_onehot_tensors(X):
    """Create tensors for MLP input."""
    x_cont = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
    x_cat = torch.zeros((len(x_cont), 0), dtype=torch.long)
    return x_cont, x_cat


# training
def train_model(model, train_pack, val_pack, *, lr=C.LR_DEFAULT,
                weight_decay=C.WEIGHT_DECAY, batch_size=C.BATCH_SIZE,
                max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE,
                device=None, verbose=False):
    """Train a model with weighted loss and early stopping.

    Returns the best validation model, training history, and best F1 score.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    (Xc_tr, Xk_tr, y_tr) = train_pack
    (Xc_va, Xk_va, y_va) = val_pack

    weights = torch.tensor(class_weights(y_tr.numpy()), dtype=torch.float32,
                           device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=max(3, patience // 3))

    loader = DataLoader(TensorDataset(Xc_tr, Xk_tr, y_tr),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    Xc_va_d, Xk_va_d = Xc_va.to(device), Xk_va.to(device)
    y_va_np = y_va.numpy()

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0

        for xb_c, xb_k, yb in loader:
            xb_c, xb_k, yb = xb_c.to(device), xb_k.to(device), yb.to(device)

            optimiser.zero_grad()
            loss = criterion(model(xb_c, xb_k), yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            running += loss.item() * len(yb)

        train_loss = running / len(y_tr)

        model.eval()
        with torch.no_grad():
            logits = model(Xc_va_d, Xk_va_d)
            val_loss = criterion(logits, y_va.to(device)).item()
            preds = logits.argmax(dim=1).cpu().numpy()

        val_f1 = evaluate(y_va_np, preds)["f1_macro"]

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_f1_macro": val_f1})

        scheduler.step(val_f1)

        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} "
                          f"(best epoch {best_epoch}, val F1={best_f1:.4f})")
                break

        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_F1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_f1


@torch.no_grad()
def predict_proba(model, x_cont, x_cat, device=None, batch_size=1024):
    """Generate prediction probabilities in batches."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    out = []

    for i in range(0, len(x_cont), batch_size):
        logits = model(x_cont[i:i + batch_size].to(device),
                       x_cat[i:i + batch_size].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(out)


def fit_continuous_scaler(train_df, continuous_cols=None) -> StandardScaler:
    return StandardScaler().fit(train_df[continuous_cols or NUMERIC])