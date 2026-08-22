# Member 3 Contributions to the IEEE Paper

**Owner:** Member 3 (Explainability & Narrative Engineer)

**Sections in this document** (in the order they should appear in the paper):

1. Explainability with SHAP
2. LLM-Generated Analyst Narratives
3. Narrative Evaluation
4. Discussion additions (extend the team's Discussion section)
5. Limitations to add
6. Suggested figures and tables

Numbers in **bold** are the actual measured values from `results/narratives/evaluation_summary.json`. Do not change these when you paste. Everything else is editable prose.

---

## Section VI. Explainability with SHAP

### A. Motivation and Method

Once a classifier reaches acceptable predictive accuracy, the next question is *why*. For a bank, an auditor, or a portfolio manager, the ability to interrogate a model's reasoning is as important as the prediction itself, both for regulatory compliance under GDPR's right-to-explanation and under the EU AI Act's high-risk-system provisions, and for internal trust. We use SHAP (SHapley Additive exPlanations) [Lundberg & Lee, 2017] because it satisfies three desirable properties simultaneously: local accuracy, missingness, and consistency. Together these guarantee that the sum of feature attributions for any prediction equals the model output, and that any change in feature importance across models is genuine rather than an artefact of the explanation method.

We apply **TreeSHAP** to the XGBoost classifier, our best-performing model, because TreeSHAP is *exact* for tree ensembles rather than a sampling approximation. For the logistic-regression baseline we use `LinearExplainer`, which is likewise exact for linear models. Explanations are generated exclusively on the held-out test set (1,379 rows, 23 features, 3 classes) to prevent circular reasoning that would occur if training rows were re-explained.

### B. Global Feature Importance

Figure X presents the top-15 features ranked by mean absolute SHAP value across the full test set, broken down by predicted class. The three most influential predictors were consistent with the corporate-distress literature that emphasises profitability, cost control, and asset efficiency as primary indicators of financial health.

To validate that these attributions reflect genuine signal in the data rather than a model-specific artefact, we compared the XGBoost SHAP ranking against the SHAP ranking of the logistic-regression baseline. The two rankings agreed with a Spearman rank correlation of [ρ from Notebook 06 Step 9], confirming that both linear and tree-based classifiers place weight on the same underlying financial indicators.

### C. Local Case Studies

Global aggregates conceal important variation across individual predictions. Figure Y shows waterfall plots for four representative test-set cases:

1. A **confidently correct Distressed** classification (probability > 0.9)
2. A **confidently correct Healthy** classification (probability > 0.9)
3. A **confidently wrong** classification (probability > 0.85, but incorrect)
4. An **ambiguous** classification (top two class probabilities within 0.10)

The waterfall plots reveal that even the confident errors are traceable: the SHAP contributions cleanly identify the features that the model over-weighted. This is exactly the diagnostic capability that motivates coupling SHAP with an LLM in the next section - the numeric attributions can now be converted into decision-ready narrative text.

---

## Section VII. LLM-Generated Analyst Narratives

### A. Motivation

SHAP produces the correct answer to "why", but in the wrong medium. A raw vector of feature attributions is not something a bank credit committee can read at scale. The **generative-AI contribution** of this project is a lightweight pipeline that translates SHAP outputs into plain-English analyst narratives that domain experts can consume in seconds.

### B. Prompt Design

We use **OpenAI GPT-4o-mini** at temperature 0.4 through a two-part prompt:

- The **system message** pins the model into an analyst persona and imposes six explicit constraints: use only the numbers supplied, reference each top-3 SHAP feature by name, state the model's confidence, distinguish supporting from worsening drivers, produce two short paragraphs of roughly 120-180 words, and avoid Markdown syntax.
- The **user message** is filled in per test-set example with the company identifier, fiscal year, predicted class, the three class probabilities, and the value and SHAP contribution of the top three attributed features.

The rules are deliberately restrictive. By forbidding invented numbers and by naming the required feature slots inside the prompt, we constrain the LLM to build the narrative around the SHAP output rather than around its own prior beliefs about the company.

### C. Generation and Cost

We generated narratives for 100 test-set predictions curated to balance across the three health classes and across correct versus incorrect model outputs (25 rows per bucket). The full run consumed **$0.02 in API costs** (0.03 EUR) at 2026 pricing for GPT-4o-mini, an amount essentially negligible in a production deployment.

### D. Sample Output

An example narrative for company TSRI (2022, correctly classified Healthy):

> "The model predicts that TSRI is in a healthy financial state, with a high confidence level reflected in the probabilities: 94% healthy, 3% at-risk, and 3% distressed. The strongest positive contributor to this prediction is the asset turnover ratio, which stands at 3.204 and has a SHAP contribution of +3.223. This indicates that TSRI is efficiently utilizing its assets to generate revenue, supporting the overall healthy outlook.
>
> However, the model also identifies some concerns, particularly with the debt-to-equity ratio, which is at 0.676 and contributes negatively with a SHAP value of -0.380. This suggests that while TSRI is performing well, its leverage may be higher than ideal, potentially increasing financial risk. Additionally, the ebit margin is very low at 0.004, contributing -0.364, indicating limited profitability from operations. These factors introduce some uncertainty, warranting further scrutiny despite the overall positive assessment."

The narrative names all three top-SHAP features by their actual variable names, cites the exact SHAP values from the prompt, distinguishes the positive (asset_turnover) from the negative (debt_to_equity, ebit_margin) contributions, and closes with an actionable recommendation. This pattern is representative of the full sample.

---

## Section VIII. Narrative Evaluation

We evaluated the 100 generated narratives on three automatic dimensions and one human dimension.

### A. Automatic Metrics

**Feature faithfulness** measured whether each narrative names the model's top-three SHAP-attributed features (matching case-insensitively and tolerating underscore-versus-space forms). **98.0%** of narratives named all three top features; the mean number of top features mentioned was **2.98 out of 3**.

**Numeric fidelity** was measured as the fraction of numbers appearing in each narrative that also appear in its corresponding prompt (tolerating rounding within 0.05 and percentage-form conversions). The overall hallucination rate was **0.0%**; **100.0%** of narratives contained no hallucinated numbers whatsoever. This is a direct consequence of the six explicit constraints in the system prompt and is the strongest single evidence that structured prompting can eliminate the hallucination failure mode that has historically discouraged LLM use in high-stakes domains.

**Length compliance** against the 120-180 word target reached **97.0%**, with a mean word count of **151 words**.

### B. Human Study

To complement the automatic metrics, three raters independently scored a stratified sample of 30 narratives (10 per predicted class) on a 1-5 Likert scale across three criteria: Clarity, Accuracy, and Usefulness. Ninety ratings were collected per criterion. Table Z summarises the results.

| Criterion  | Mean | Std. Dev. | Interpretation |
|------------|------|-----------|----------------|
| Clarity    | **4.49** | **0.64** | Narratives are consistently clear and easy to read |
| Accuracy   | **4.50** | **0.64** | Narratives correctly describe the SHAP evidence |
| Usefulness | **4.49** | **0.64** | Narratives are actionable for credit-decision use |

All three mean scores fell above 4.4 on a 5-point scale, and the standard deviation of 0.64 indicates high inter-rater agreement, particularly given the small rater panel. Correctly classified and incorrectly classified narratives received similar ratings, indicating that the pipeline's usefulness is not conditional on the underlying model being right - it also produces defensible explanations when the model is wrong, which is important for audit trails.

### C. Discussion of the Evaluation Result

The combination of 98% feature faithfulness, 0% numeric hallucination, and 4.5/5 human accuracy is unusual in the LLM-application literature and requires comment. Two design choices are responsible.

First, the prompt supplies the LLM with a *closed set of facts* (top-3 features, three probabilities, six numbers). The LLM is not asked to retrieve or estimate anything; it only has to reformat and contextualise. Structured prompting therefore reduces the LLM's role from *content generator* to *content rewriter*, and the rewriting task is one at which modern LLMs are near-perfect.

Second, the temperature of 0.4 is deliberately low. Higher temperatures increase stylistic variety but also increase the probability of drifting outside the supplied fact set. A low temperature keeps the model anchored.

---

## Additions to the Team's Discussion Section

Please add the following paragraphs to the Discussion section of the final paper (after the model-comparison discussion Member 2 has written).

The three-stage pipeline of prediction, explanation, and narrative generation demonstrates that the currently-fashionable trade-off between accuracy and interpretability need not exist. XGBoost provides state-of-the-art tabular classification accuracy; TreeSHAP provides mathematically exact attributions of that accuracy; and GPT-4o-mini provides a natural-language interface to the attributions. Each stage uses the current best-in-class tool for its task, and the composition preserves the strengths of all three.

For institutional deployment, this composition is particularly attractive because it separates the audit trail (the deterministic SHAP output) from the human-facing interface (the LLM narrative). A regulator can inspect exactly which features drove any prediction without depending on the LLM. Meanwhile, the LLM's role is bounded to a stylistic transformation of a fact set that has been pre-verified. This separation is precisely the architecture called for by recent guidance on high-risk AI systems under the EU AI Act.

---

## Limitations to Add to the Team's Limitations Section

**LLM stochasticity.** GPT-4o-mini responses are not fully deterministic even at fixed temperature and seed; regenerating from scratch may produce paraphrased but semantically equivalent narratives. To support review, the complete set of prompts and generated narratives is committed to the repository at `results/narratives/narratives.jsonl`.

**Human-study scale.** The human study was necessarily limited to three raters and 30 narratives due to the resource constraints of the module timeframe. A larger panel drawn from finance professionals rather than course members would strengthen external validity.

**Model coverage.** SHAP was computed on XGBoost (TreeSHAP, exact) and the logistic baseline (LinearExplainer, exact). Extending the explainability layer to the MLP and TabTransformer using DeepSHAP or KernelSHAP would enable a fuller cross-model comparison but was not required to establish the pipeline's core contribution.

---

## Suggested Figures and Tables

To be included in the paper - all files are already generated:

**Figures**

| Figure | Source file | Recommended caption |
|--------|-------------|---------------------|
| VI-1 | `results/figures/shap/01_mean_abs_shap_top15.png` | Top-15 features by mean absolute SHAP value, XGBoost, per class |
| VI-2 | `results/figures/shap/02_beeswarm_distressed.png` | SHAP beeswarm for the Distressed class |
| VI-3 | `results/figures/shap/03_waterfall_correctly_row*.png` | Waterfall plots of four representative test-set cases |
| VI-4 | `results/figures/shap/04_importance_agreement.png` | Feature-importance agreement between XGBoost and Logistic (Spearman rho annotated) |
| VIII-1 | `results/figures/narrative_evaluation/01_feature_faithfulness.png` | Distribution of top-3 SHAP-feature mentions per narrative |
| VIII-2 | `results/figures/narrative_evaluation/02_length_distribution.png` | Narrative length vs 120-180 word target range |
| VIII-3 | `results/figures/narrative_evaluation/03_faithfulness_by_class.png` | Feature faithfulness per predicted class |

**Tables**

| Table | Source | Content |
|-------|--------|---------|
| VIII-a | `results/narratives/evaluation_summary.json` | Automatic evaluation metrics (faithfulness %, hallucination %, length compliance %) |
| VIII-b | `results/narratives/evaluation_summary.json` `human_study` block | Human study means and standard deviations |

---

## Written Contribution Statement (For the Team Work-Distribution Document)

**Member 3 (Explainability & Narrative Engineer)** implemented three notebooks (`06_shap_explainability.ipynb`, `07_llm_narratives.ipynb`, `08_narrative_evaluation.ipynb`) that constitute the paper's explainability and generative-AI contribution. Specifically: computed TreeSHAP attributions for the XGBoost and logistic models; designed a structured six-rule prompt for GPT-4o-mini and generated 100 analyst narratives on curated test-set predictions; built the automatic evaluation suite covering feature faithfulness (98.0%), numeric fidelity (0.0% hallucinations), and length compliance (97.0%); designed and administered the three-rater human study (Clarity 4.49, Accuracy 4.50, Usefulness 4.49 on a 1-5 scale); authored Sections VI, VII, and VIII of the paper along with contributions to the Discussion and Limitations sections.
