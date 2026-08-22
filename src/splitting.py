"""Company-level stratified splitting. Member 2, H9DLGA.

Splitting by ROW leaks: 76% of companies carry one label across all
their years, so the same company in train and test = memorisation.
We split the COMPANY list, stratified by each company's dominant label.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_company_splits(df, seed=42, val_size=0.15, test_size=0.15):
    dom = (df.groupby('company_id')['label']
             .agg(lambda s: s.value_counts().idxmax()))
    companies = dom.index.to_numpy()
    strat = dom.to_numpy()

    tr_c, hold_c, tr_s, hold_s = train_test_split(
        companies, strat, test_size=val_size + test_size,
        random_state=seed, stratify=strat)

    rel = test_size / (val_size + test_size)
    val_c, test_c = train_test_split(
        hold_c, test_size=rel, random_state=seed, stratify=hold_s)

    return {'train': sorted(tr_c.tolist()),
            'val': sorted(val_c.tolist()),
            'test': sorted(test_c.tolist())}


def save_splits(splits, path, seed):
    payload = {'seed': seed,
               'n_train_companies': len(splits['train']),
               'n_val_companies': len(splits['val']),
               'n_test_companies': len(splits['test']),
               **splits}
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_splits(path):
    with open(path) as f:
        d = json.load(f)
    return {k: d[k] for k in ('train', 'val', 'test')}


def apply_splits(df, splits):
    return (df[df.company_id.isin(splits['train'])].copy(),
            df[df.company_id.isin(splits['val'])].copy(),
            df[df.company_id.isin(splits['test'])].copy())


def assert_no_overlap(splits):
    a, b, c = (set(splits['train']), set(splits['val']), set(splits['test']))
    assert not (a & b), 'train/val company overlap'
    assert not (a & c), 'train/test company overlap'
    assert not (b & c), 'val/test company overlap'
    return True
