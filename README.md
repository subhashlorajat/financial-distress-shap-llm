# Explainable Financial Health Scoring

Predicts company financial distress from annual filings, then explains every prediction
in plain English — combining gradient boosting, TreeSHAP attributions, and an LLM
narrative layer whose output is automatically checked for factual faithfulness.

MSc Data Analytics group project (H9DLGA), National College of Ireland, 2026.
Submitted as an IEEE-format paper — see [`report.pdf`](report.pdf).

---

## Key results

| | |
|---|---|
| Dataset | **9,284 company-years**, 2,942 US companies, FY2021–2025 (SimFin, parsed from SEC 10-K XBRL) |
| Task | 3-class distress classification (Healthy / At-Risk / Distressed), Altman Z'-Score labels |
| Best model | **XGBoost — 0.847 F1-macro**, 0.963 AUC-ROC |
| Explanation faithfulness | **98.0%** of narratives name all top-3 SHAP features |
| Numeric hallucination rate | **0.0%** across 100 generated narratives |
| Human evaluation | **4.50 / 5** accuracy, 3 raters, 90 ratings |
| Cost to generate 100 narratives | **$0.016** |

**Headline finding:** deep tabular models did not beat gradient boosting here.
TabTransformer trailed XGBoost by 0.093 F1-macro — roughly 8× the run-to-run standard
deviation across five seeds — while training about 470× slower (281s vs 0.6s per seed).

---

## My contribution

This was a three-person project. I was **Member 3 — Explainability & Narrative Engineer**,
responsible for Steps 7–9 of the pipeline. Specifically, I:

- Computed **TreeSHAP** attributions for the XGBoost model and LinearExplainer attributions
  for the logistic baseline, across the full 1,379-row test set
- Designed a **structured six-rule prompt** for GPT-4o-mini that converts SHAP output into
  an analyst-style narrative, constrained to cite only the model's actual drivers
- Generated and published **100 narratives** on curated test-set predictions
- Built the **automatic evaluation suite** measuring feature faithfulness, numeric
  hallucination and length compliance — the numbers in the table above
- Designed and administered a **three-rater human study** (Clarity 4.49, Accuracy 4.50,
  Usefulness 4.49 on a 1–5 scale, 90 ratings total)
- Authored paper Sections VI–VIII plus Discussion and Limitations contributions

Teammates **Aditya Tomar** built the data pipeline and labelling (Steps 1–3), and
**Smit Gandubhai Savaliya** trained and compared the four classifiers (Steps 4–6).
Their full handover notes are preserved in [`docs/team-deliverables.md`](docs/team-deliverables.md).

---

## Why the explanation layer matters

A distress score on its own is not actionable — a credit analyst needs to know *why*.
The risk with using an LLM for that is fabrication: a fluent narrative that cites numbers
the model never used. So the narrative layer is treated as a component to be measured,
not trusted:

1. **TreeSHAP** identifies the top-3 drivers for each individual prediction
2. A **constrained prompt** passes only those drivers and their real values to the LLM
3. An **automatic checker** verifies every narrative names the actual top-3 features and
   invents no numbers absent from the input
4. A **human study** independently rates clarity, accuracy and usefulness

Result: 98% faithfulness and zero numeric hallucinations across 100 narratives.

---

## Repository structure

```
src/                    data pipeline, feature engineering, model training
notebooks/
  01–02                 data exploration and company-level splitting
  03–05                 baseline models, TabTransformer, model comparison
  06_shap_explainability.ipynb    TreeSHAP on XGBoost + logistic baseline   <- mine
  07_llm_narratives.ipynb         100 narratives via GPT-4o-mini            <- mine
  08_narrative_evaluation.ipynb   automatic metrics + human study           <- mine
results/
  shap/                 SHAP value arrays, top-feature tables, confident errors
  narratives/           narratives.jsonl, faithfulness scores, human_study.xlsx
  figures/              model comparison, SHAP plots, evaluation plots
models/                 trained XGBoost, logistic, MLP, TabTransformer artefacts
paper/                  per-section drafts
docs/team-deliverables.md   original three-member handover documentation
report.pdf              the submitted IEEE-format paper
```

---

## Reproducing the results

```bash
pip install -r requirements.txt
```

**Model training** (`data/splits.json` is committed and must not be regenerated —
every reported number depends on it):

```bash
python run_all.py --skip-prep     # full pipeline on the committed splits
python run_all.py --quick         # smoke test, ~3 min
```

**Explainability and evaluation** — notebooks 06 and 08 need no API key; they read
models and predictions already committed:

```bash
jupyter nbconvert --to notebook --execute notebooks/06_shap_explainability.ipynb
jupyter nbconvert --to notebook --execute notebooks/08_narrative_evaluation.ipynb
```

**Narrative generation** (notebook 07) requires an OpenAI API key. The generated
narratives are already committed to `results/narratives/`, so a key is only needed to
regenerate them. Place it in `openai_api_key.txt` (gitignored). Full regeneration costs
about $0.02.

---

## Limitations

- Altman Z'-Score labels are a **proxy** for distress, not observed bankruptcy outcomes.
  The model learns the Z'-Score boundary as much as distress itself.
- Financial-sector firms are excluded (no COGS line item), standard in the distress
  literature but a real coverage gap.
- The dataset covers FY2021–2025 only, a period shaped by post-pandemic conditions.
  Generalisation to other macro regimes is untested.
- Faithfulness checks confirm the narrative cites the model's real drivers. They do not
  confirm the model's reasoning is economically correct.
- The human study used three raters on 30 narratives — indicative, not conclusive.

---

## Stack

Python · XGBoost · scikit-learn · PyTorch · SHAP · Optuna · OpenAI API · pandas
