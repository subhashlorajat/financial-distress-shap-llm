# Member 1 — Data Engineer deliverables (H9DLGA Group Project)

## What this contains
- `src/data_pipeline.py`   — Steps 1–3: parse SimFin statements, compute the 4 ratios, Altman Z' labels
- `src/eda.py`             — generates the 5 EDA plots in `results/plots/`
- `notebooks/01_data_exploration.ipynb` — hand-off notebook (per project guide §4.1)
- `data/labelled.csv`      — HAND-OFF FILE for Members 2 & 3
                             columns: company_id, year, gross_margin, net_margin,
                             revenue_growth, cost_growth, label
- `data/labelled_full.csv` — same rows plus raw line items and z_score (useful for SHAP context)
- `paper/data_quality_section.md` — ~500-word data-quality draft for the paper
- `results/plots/`         — figures for the paper

## How to reproduce
1. Python 3.10+, then: `pip install -r requirements.txt`
2. Place `us-income-annual.csv` and `us-balance-annual.csv` in `raw/`
   (and point RAW_DIR in `src/data_pipeline.py` at that folder)
3. `python src/data_pipeline.py` then `python src/eda.py`

## Key decisions to mention in the paper
- Data source: SimFin bulk annual statements (parsed from SEC 10-K XBRL) instead of
  scraping EDGAR directly — same regulatory data, fewer parsing errors.
- Financial firms (no COGS) excluded — standard in distress-prediction literature.
- Altman Z' (private-firm, book-equity) variant used because no market prices are
  available. Zones: >2.90 Healthy, 1.23–2.90 At-Risk, <1.23 Distressed.
- Ratios winsorised at 1%/99%. Growth ratios require consecutive fiscal years.
- Final dataset: 9,284 company-years, 2,942 companies, FY2021–2025.
  Class split: 44.6% Distressed / 40.5% At-Risk / 14.8% Healthy.

<!------------------------------------------>

# Member 2 — Model Engineer deliverables (Smit Savaliya)

Covers **Steps 4–6** of the project workflow: company-level splitting, four
trained classifiers, the model comparison, and the hand-off package for
Member 3.

## What this contains

```
src/config.py             single source of truth for paths, seeds, constants
src/features.py           feature engineering + leakage-safe preprocessing
src/splitting.py          company-level stratified splitting
src/data_prep.py          Step 4 — builds splits.json and all hand-off artefacts
src/evaluation.py         shared metrics, experiment logging, class weights
src/nn_models.py          MLP and TabTransformer (PyTorch, Huang et al. 2020)
src/nn_trainer.py         shared training loop — identical protocol for both
src/train_baselines.py    Step 5 — Logistic Regression, XGBoost
src/train_neural.py       Steps 5–6 — MLP, TabTransformer
src/build_report.py       comparison tables + 5 figures
src/predict.py            HAND-OFF ENTRY POINT for Member 3
src/validate_handoff.py   automated pre-hand-off checks

notebooks/02_data_splitting.ipynb     preprocessing decisions, documented
notebooks/03_baseline_models.ipynb    Logistic Regression + XGBoost
notebooks/04_tabtransformer.ipynb     MLP + TabTransformer, RQ1
notebooks/05_model_comparison.ipynb   final comparison + hand-off

run_all.py                one-command pipeline
requirements.txt          pinned versions (SHAP is version-sensitive)
MEMBER3_GUIDE.md          complete brief for the explainability stage
```

## How to reproduce

```bash
pip install -r requirements.txt

python run_all.py --skip-prep     # reuse the committed splits.json
python run_all.py --quick         # smoke test, ~3 min
```

Individual stages:

```bash
python src/data_prep.py                      # Step 4 — run ONCE only
python src/train_baselines.py --trials 40    # Step 5
python src/train_neural.py --trials 25       # Steps 5–6 (~52 min on CPU)
python src/build_report.py                   # tables + figures
python src/validate_handoff.py               # pre-hand-off gate
```

`data/splits.json` is committed and must **not** be regenerated — every
reported number depends on it. Always use `--skip-prep`.

## Results

Test set, mean ± standard deviation across five seeds (42, 7, 123, 2024, 999).
Tuning used 40 Optuna trials for XGBoost and 25 for each neural model.

| Model | F1-macro | AUC-ROC | F1 At-Risk | F1 Distressed | F1 Healthy | Train (s) | Params |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.737 ± 0.000 | 0.907 | 0.718 | 0.854 | 0.638 | 0.0 | 72 |
| **XGBoost** | **0.847 ± 0.001** | **0.963** | **0.835** | **0.897** | **0.810** | 0.6 | 2,156,224 |
| MLP | 0.789 ± 0.015 | 0.941 | 0.778 | 0.877 | 0.713 | 8.1 | 40,195 |
| TabTransformer | 0.754 ± 0.011 | 0.932 | 0.737 | 0.862 | 0.664 | 281.4 | 65,431 |

Per-seed test F1-macro:

| Model | 42 | 7 | 123 | 2024 | 999 |
|---|---|---|---|---|---|
| XGBoost | 0.8473 | 0.8478 | 0.8489 | 0.8467 | 0.8460 |
| MLP | 0.7883 | 0.8030 | 0.7955 | 0.7648 | 0.7953 |
| TabTransformer | 0.7424 | 0.7578 | 0.7608 | 0.7421 | 0.7669 |

**RQ1 — do deep tabular models outperform gradient boosting?**
No, not on this dataset. TabTransformer trails XGBoost by **0.093 F1-macro**
against a pooled standard deviation of **0.011**, so the gap is roughly **8×**
the observed run-to-run variation and is not attributable to sampling noise.
XGBoost also trains approximately **470× faster** (0.6 s versus 281.4 s per
seed), a material consideration for periodic retraining on updated filings.

The specific failure mode is worth reporting. On the best-validation seed,
TabTransformer reaches Healthy recall of 0.831 but Healthy precision of only
0.559, against 0.897 / 0.735 for XGBoost. It systematically over-predicts the
minority class, and those false positives are drawn mainly from At-Risk, whose
recall falls to 0.694 versus 0.827 for XGBoost. With only four categorical
tokens per row, self-attention has limited relational structure to exploit while
the extra parameters increase susceptibility to overfitting under class-weighted
loss.

Notably TabTransformer also underperforms the simpler MLP (0.754 versus 0.789),
indicating the additional capacity is not merely unhelpful but actively harmful
at this data scale.

## Fixes applied to the Member 1 hand-off

Member 1's pipeline is sound and well documented — the gap was in the original
project guide, which specified four income-statement ratios without checking
they could support balance-sheet-derived labels. All seven fixes are implemented
inside Member 2's code; nothing was sent back.

| # | Issue | Resolution |
|---|---|---|
| 1 | `z_score` is a direct label leak (label is a pure threshold on it) | excluded via `config.LEAK_COLS` |
| 2 | Altman Z′ reconstructs exactly from raw columns (r = 1.000000, max diff = 0.0) | exact X1–X5 excluded; conceptually related ratios used instead |
| 3 | No categorical features → TabTransformer degenerates to an MLP, RQ1 unanswerable | four categoricals derived, bin edges fitted on train only |
| 4 | Negative book equity in 711 rows (7.7%) breaks equity-denominated ratios | explicit `negative_equity` flag + NaN-masked denominator |
| 5 | FY2025 is a partial filing season (126 rows, early filers skew healthy) | dropped in `load_raw()` |
| 6 | Preprocessing constants must not see validation/test | `fit_preprocessor(train_df)` learns everything; `apply_preprocessor` replays |
| 7 | 76.2% of companies carry one label across all years → row splits leak | company-level stratified split, `assert_no_overlap()` |

### Why `labelled_full.csv` and not `labelled.csv`

The four hand-off ratios are all income-statement quantities; Altman Z′ labels
are computed almost entirely from balance-sheet quantities. Measured on a tuned
Random Forest with correct company-level splitting:

| Feature set | F1-macro | Healthy F1 | Healthy recall |
|---|---|---|---|
| Original 4 ratios | 0.490 | 0.159 | 9.6 % |
| After fixes above | **0.847** | **0.810** | **89.7 %** |

On the original four ratios the model identified roughly one healthy company in
ten. The information required to separate the classes was not present in the
feature space.

### Circularity audit — result and interpretation

Because the labels derive from a formula exactly reconstructible from the raw
data, excluding the Altman components requires empirical verification rather
than assertion.

**Top feature `profit_regime_loss` carries 54.4% of total XGBoost gain.** This
is below the 60% threshold at which circularity would be suspected, so the audit
passes, but the concentration is high enough to warrant explicit discussion in
the paper rather than a passing mention.

Two points support the reading that this reflects genuine economic signal rather
than label leakage:

1. `profit_regime` is derived from `net_margin`, an income-statement ratio. It
   is **not** one of the five Altman Z′ components, all of which are
   balance-sheet quantities scaled by total assets. There is no algebraic path
   from this feature to the label.
2. The distress-prediction literature consistently identifies sustained
   loss-making as the dominant single predictor of financial distress. A model
   ranking it first is expected, not suspicious.

The concentration is also hyperparameter-sensitive: an earlier configuration
with more estimators and weaker `min_child_weight` produced 28.9% for the same
feature. XGBoost gain importance is known to concentrate under shallower, more
strongly regularised trees. A permutation-importance cross-check is recommended
as a robustness statement, since permutation importance is less biased than gain
for this purpose.

## Key decisions to mention in the paper

- **Labels are a proxy, not ground truth.** Altman Z′ zones stand in for actual
  distress events, so small unprofitable listed firms land in the Distressed
  zone without ever filing Chapter 11. This is the single largest limitation and
  is stated openly in the Evaluation section.
- **Healthy is the minority class (14.8%), not Distressed.** This inverts the
  usual cost asymmetry: the expensive error here is flagging a healthy company
  as distressed, the opposite of a fraud-detection framing.
- **F1-macro is the headline metric, never accuracy.** A model that never
  predicts Healthy still scores ~45% accuracy.
- **Feature exclusion is deliberate and documented.** Exact Altman components
  were withheld to avoid circularity; `current_ratio`, `debt_to_equity`,
  `opex_ratio`, `ebit_margin` and `asset_turnover` overlap conceptually without
  reproducing the formula.
- **Both neural models share one training loop** (AdamW, class-weighted
  cross-entropy, gradient clipping at 5.0, early stopping on validation
  F1-macro), so RQ1 isolates architecture rather than training budget.
- **Model selection is always on validation.** The test fold was evaluated once,
  after all design decisions were finalised.
- **Five seeds, mean ± std.** Observed noise is 0.001 (XGBoost) to 0.015 (MLP)
  F1-macro, so gaps below roughly 0.03 for the neural models are reported as
  indistinguishable.
- **Known inherited limitation:** Member 1 winsorised the original four ratios
  on the full dataset before hand-off, so those four columns carry a mild
  global-statistics leak that cannot be undone downstream. All newly engineered
  features are clipped train-only. Disclosed rather than hidden.
- **TabTransformer implemented directly in PyTorch** from Huang et al. (2020)
  rather than via a wrapper, so the architecture can be described exactly in the
  paper and reproduction has one fewer dependency.

## Hand-off to Member 3

`python src/validate_handoff.py` — **23 passed, 0 failures**, 1 warning
(`shap` not yet installed in the Member 2 environment; Member 3 installs it).

| File | Purpose |
|---|---|
| `data/splits.json` | company IDs per fold — evaluate on the identical test set |
| `models/preprocessor.joblib` | replay identical preprocessing on any row |
| `models/label_encoder.joblib` | class index ↔ name mapping |
| `models/feature_columns.json` | exact column order — SHAP breaks if it drifts |
| `models/xgboost_model.json` | best model checkpoint (explain this one) |
| `models/mlp_model.pt`, `models/tabtransformer_model.pt` | neural checkpoints |
| `results/background_sample.csv` | 100 training rows for DeepSHAP baselines |
| `results/test_predictions.csv` | per-row predictions and confidences |
| `results/confident_errors.csv` | high-confidence mistakes — best SHAP case studies |
| `src/predict.py` | single entry point |
| `MEMBER3_GUIDE.md` | full brief for the explainability stage |

### Worked example

```python
import sys; sys.path.insert(0, "src")
from predict import explain_ready
import shap, numpy as np, pandas as pd

pkg = explain_ready("xgboost")          # model + test set + matrix + background

explainer = shap.TreeExplainer(pkg["model"])
shap_values = np.asarray(explainer.shap_values(pkg["X"]))   # (rows, features, classes)

i = 0
pred_idx = pkg["y_pred"][i]
top = (pd.Series(shap_values[i, :, pred_idx], index=pkg["feature_columns"])
         .sort_values(key=abs, ascending=False).head(3))

print(pkg["y_pred_labels"][i], pkg["proba"][i].max())
print(top)          # -> feed these into the LLM prompt
```

### Rules for Member 3

- **Always enter through `predict.py`.** Reading the CSV directly bypasses the
  FY2025 exclusion and the fitted preprocessing.
- **Never re-derive medians or bin edges.** `preprocessor.joblib` is the single
  source of truth.
- **Never regenerate `splits.json`.** Every reported number depends on it.
- **Install the pinned versions in `requirements.txt` before adding SHAP.**
  TreeSHAP is sensitive to `scikit-learn` and `xgboost` mismatches.
- **`results/confident_errors.csv` is the best starting point** for narrative
  case studies — a confident correct prediction and a confident wrong one
  produce very different analyst reports.

## Reproducibility notes

- Split seed `42` is written into `splits.json` and must not change.
- Model seeds `[42, 7, 123, 2024, 999]` are set across Python, NumPy and PyTorch;
  `cudnn.deterministic` is enabled so results reproduce on GPU machines.
- Every run appends to `results/experiment_log.csv` (model, seed,
  hyperparameters, val/test F1, wall-clock time, parameter count).
- Selected XGBoost hyperparameters: `n_estimators` 300, `max_depth` 6,
  `learning_rate` 0.0311, `subsample` 0.805, `colsample_bytree` 0.862,
  `min_child_weight` 10, `reg_lambda` 0.805, `reg_alpha` 0.00085.
- Total pipeline runtime approximately 55 minutes on CPU, dominated by
  TabTransformer tuning and its five final seeds.

<!------------------------------------------>

# Member 3 — Explainability & Narrative Engineer deliverables (H9DLGA Group Project)

Covers **Steps 7–9** of the project workflow: SHAP explainability, LLM-based
analyst narrative generation, and evaluation.

## What this contains

```
notebooks/06_shap_explainability.ipynb   TreeSHAP on XGBoost + logistic baseline
notebooks/07_llm_narratives.ipynb        100 narratives via GPT-4o-mini
notebooks/08_narrative_evaluation.ipynb  Auto metrics + human-study spreadsheet

results/shap/
  shap_values_xgboost.npy         (1379, 23, 3) SHAP array for the test set
  expected_values_xgboost.npy     class base values
  shap_values_logistic.npy        Linear-SHAP for the baseline comparison
  shap_top_features_xgboost.csv   tidy per-row top-3 features (feeds notebook 07)
  shap_confident_errors.csv       high-confidence mistakes with SHAP context

results/narratives/
  narratives.jsonl                100 prompts + narratives + token usage
  narratives.csv                  same, tabular
  faithfulness_scores.csv         per-row automatic metric scores
  evaluation_summary.json         headline numbers for the paper
  human_study.xlsx                three-rater Likert spreadsheet (30 rows)

results/figures/shap/             8 SHAP plots referenced in Section VI
results/figures/narrative_evaluation/   3 evaluation plots for Section VIII

paper/member3_sections.md         ready-to-paste Sections VI, VII, VIII plus
                                  Discussion additions and Limitations
```

## How to reproduce

Notebook 06 (SHAP) and Notebook 08 (evaluation) run without external API
access. They read models and predictions that Member 2 already committed:

```bash
# assumes requirements.txt is installed and the .venv is active
jupyter nbconvert --to notebook --execute notebooks/06_shap_explainability.ipynb
jupyter nbconvert --to notebook --execute notebooks/08_narrative_evaluation.ipynb
```

Notebook 07 (LLM narratives) requires an OpenAI API key. **The generated
narratives are already committed to `results/narratives/`, so evaluators do
not need a key to inspect them.** To regenerate from scratch:

```bash
# 1. Create openai_api_key.txt in the project root (gitignored)
# 2. Then run:
jupyter nbconvert --to notebook --execute notebooks/07_llm_narratives.ipynb
```

Cost of a full regeneration: approximately $0.02 (0.03 EUR).

## Results

Automatic metrics on 100 LLM-generated narratives (GPT-4o-mini, temperature 0.4):

| Metric | Value |
|---|---|
| Feature faithfulness (all top-3 SHAP features named) | **98.0 %** |
| Mean top-3 features mentioned | **2.98 / 3** |
| Numeric hallucination rate | **0.0 %** |
| Narratives with zero hallucinated numbers | **100.0 %** |
| Length compliance (120–180 words) | **97.0 %** |
| Mean word count | 151 |
| Total API cost | **$0.016** |

Human study on a stratified sample of 30 narratives (three raters, 1–5 Likert,
90 ratings per criterion):

| Criterion | Mean | Std |
|---|---|---|
| Clarity | **4.49** | 0.64 |
| Accuracy | **4.50** | 0.64 |
| Usefulness | **4.49** | 0.64 |

## Key decisions to mention in the paper

- **TreeSHAP for XGBoost, LinearExplainer for Logistic Regression.** Both are
  exact (not sampling approximations). DeepSHAP for the neural models was
  planned but not required to establish the pipeline’s contribution.
- **Explanations on the test set only.** Explaining training rows would be
  circular.
- **Prompt is structured, not open-ended.** The system message enforces six
  rules (use only supplied numbers, name every top-3 feature, state confidence,
  distinguish supporting from worsening drivers, 120–180 words, no Markdown).
  These constraints, combined with a low temperature (0.4), produced 0 %
  numeric hallucination across 100 generations.
- **Sample of 100 narratives is deliberately balanced.** 25 correct-Healthy,
  25 correct-At-Risk, 25 correct-Distressed, 25 wrong (any class). A random
  sample would have been dominated by Distressed rows and by correct
  predictions.
- **Three-stage architecture separates audit trail from user interface.** The
  deterministic SHAP output is the audit trail; the stochastic LLM narrative
  is the user interface. Regulators can inspect SHAP directly without depending
  on the LLM.

## Rules that were followed

- All access to models and features went through `src/predict.py`, never
  directly through `labelled_full.csv`.
- `data/splits.json` was not modified; all reported numbers use Member 2’s
  committed split.
- Only the test set was explained; training rows were never inspected.
- The pinned versions in `requirements.txt` were installed before adding
  `shap`, `openai`, and `openpyxl`.