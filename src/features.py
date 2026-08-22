"""Feature engineering and preprocessing utilities. Member 2"""
import numpy as np
import pandas as pd

LEAK_COLS = ['z_score', 'label', 'company_id', 'year']
BASE_RATIOS = ['gross_margin', 'net_margin', 'revenue_growth', 'cost_growth']
ENGINEERED = ['current_ratio', 'debt_to_equity', 'opex_ratio',
              'ebit_margin', 'asset_turnover', 'negative_equity']
CATEGORICALS = ['size_bucket', 'profit_regime', 'leverage_tier', 'growth_stage']
NUMERIC = BASE_RATIOS + ENGINEERED


def load_raw(path):
    df = pd.read_csv(path)
    df = df[df.year != 2025].copy()
    return df.reset_index(drop=True)


def add_numeric_features(df):
    d = df.copy()
    d['current_ratio'] = d.curr_assets / d.curr_liab
    d['opex_ratio'] = d.opex / d.revenue
    d['ebit_margin'] = d.ebit / d.revenue
    d['asset_turnover'] = d.revenue / d.total_assets
    d['negative_equity'] = (d.book_equity <= 0).astype(int)
    safe_eq = d.book_equity.where(d.book_equity > 0, np.nan)
    d['debt_to_equity'] = d.total_liab / safe_eq
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def fit_preprocessor(train_df, q_low=0.01, q_high=0.99):
    p = {}
    p['clip'] = {c: (float(train_df[c].quantile(q_low)),
                     float(train_df[c].quantile(q_high)))
                 for c in NUMERIC if c != 'negative_equity'}
    p['median'] = {c: float(train_df[c].median()) for c in NUMERIC}
    p['size_edges'] = [float(x) for x in train_df.revenue.quantile([0, .25, .5, .75, 1.0])]
    lev = train_df.total_liab / train_df.total_assets
    p['lev_edges'] = [float(x) for x in lev.quantile([0, 1/3, 2/3, 1.0])]
    p['profit_edges'] = [-np.inf, -0.05, 0.05, np.inf]
    p['growth_edges'] = [-np.inf, 0.0, 0.15, np.inf]
    return p


def _safe_cut(series, edges, labels):
    e = list(edges)
    e[0], e[-1] = -np.inf, np.inf
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = e[i - 1] + 1e-9
    return pd.cut(series, bins=e, labels=labels, include_lowest=True)


def apply_preprocessor(df, p):
    d = df.copy()
    for c, (lo, hi) in p['clip'].items():
        d[c] = d[c].clip(lo, hi)
    for c in NUMERIC:
        d[c] = d[c].fillna(p['median'][c])
    d['size_bucket'] = _safe_cut(d.revenue, p['size_edges'],
                                 ['micro', 'small', 'mid', 'large'])
    d['leverage_tier'] = _safe_cut(d.total_liab / d.total_assets, p['lev_edges'],
                                   ['low', 'med', 'high'])
    d['profit_regime'] = _safe_cut(d.net_margin, p['profit_edges'],
                                   ['loss', 'breakeven', 'profitable'])
    d['growth_stage'] = _safe_cut(d.revenue_growth, p['growth_edges'],
                                  ['shrinking', 'stable', 'high_growth'])
    for c in CATEGORICALS:
        d[c] = d[c].astype(str)
    return d