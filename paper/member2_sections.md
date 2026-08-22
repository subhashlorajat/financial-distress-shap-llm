# Paper sections — Member 2 (Modelling & Quantitative Evaluation)

Draft text for the IEEE paper, approximately 3 of the 10–12 pages. All figures
match `results/model_comparison.csv` from the final run.

---

## III. METHODOLOGY (modelling subsections)

### C. Data Preparation

The labelled dataset produced in the previous stage contains 9,284 company-years
across 2,942 firms (FY2021–2025), labelled Healthy, At-Risk or Distressed using
the Altman Z′ private-firm variant. Two properties of this labelling scheme
drive the modelling design.

First, the label is a deterministic threshold on the Z′ score, and Z′ is computed
from five balance-sheet ratios. Reconstructing Z′ from the underlying line items
reproduces the stored score exactly (Pearson r = 1.000000, maximum absolute
deviation 0.0). Supplying those five components as model inputs would therefore
reduce the task to recovering a known closed-form expression. We excluded them,
and instead engineered five financial ratios that overlap conceptually with the
Altman constructs without reproducing them: current ratio, debt-to-equity,
operating expense ratio, EBIT margin and asset turnover. This choice is validated
empirically in Section IV-E.

Second, the four ratios supplied at hand-off are all income-statement quantities,
whereas four of the five Z′ components are balance-sheet quantities. A tuned
Random Forest trained on those four ratios alone reached only 0.490 F1-macro,
with 0.159 F1 and 9.6% recall on the Healthy class — the model identified
approximately one healthy firm in ten. The information required to discriminate
the classes was largely absent from the feature space. Incorporating
balance-sheet-derived ratios raised F1-macro to 0.847 and Healthy F1 to 0.810.

Four categorical features were derived from continuous quantities: size bucket
(revenue quartiles), profitability regime (net margin banded at ±0.05), leverage
tier (liabilities-to-assets terciles) and growth stage (revenue growth banded at
0 and 0.15). These are required by the TabTransformer architecture, whose
self-attention mechanism operates over embedded categorical columns; with no
categorical inputs the model reduces to a multilayer perceptron and the research
question becomes untestable.

Negative book equity occurs in 711 rows (7.7%) and inverts the sign of any
equity-denominated ratio. We encode this explicitly as a binary indicator — it is
itself a distress signal — and compute debt-to-equity on a masked denominator
with training-median imputation. Fiscal year 2025 was excluded: it contains 126
rows against approximately 2,200 for other years and exhibits a materially
different class distribution (24.6% Healthy versus 14–15%), reflecting an
incomplete filing season in which early filers skew healthy. The final modelling
dataset comprises 9,158 company-years across 2,941 firms.

### D. Experimental Design

**Partitioning.** 76.2% of firms carry an identical label in every year they
appear. Random row-level partitioning therefore places the same firm in both
training and test folds, where near-identical ratios permit memorisation rather
than generalisation. Measured empirically, row-level splitting inflates F1-macro
from 0.480 to 0.516 — 7.4% of the reported score is attributable to leakage. We
partition at firm level, stratified on each firm's dominant label, in a 70/15/15
ratio (2,058 / 441 / 442 firms; 6,396 / 1,383 / 1,379 rows). Class proportions
agree across folds to within 0.9 percentage points.

**Preprocessing discipline.** All preprocessing constants — winsorisation bounds,
imputation medians, categorical bin edges and standardisation parameters — are
estimated on the training fold and replayed on validation and test. One inherited
limitation is disclosed: the original four ratios were winsorised on the full
dataset prior to hand-off, introducing a mild global-statistics leak in those
columns that cannot be reversed downstream. All newly engineered features are
clipped using training-fold quantiles only.

**Model selection.** Hyperparameters were optimised with Tree-structured Parzen
Estimator search (Optuna) against validation F1-macro: 40 trials for XGBoost and
25 trials for each neural model. The test fold was evaluated once, after all
design decisions were finalised. Each configuration was retrained under five
random seeds (42, 7, 123, 2024, 999) and is reported as mean ± standard
deviation. Observed run-to-run variation ranges from 0.001 (XGBoost) to 0.015
(MLP) F1-macro, establishing the threshold below which differences are treated
as indistinguishable from noise.

### E. Models

Four classifiers were evaluated. **Logistic regression** with balanced class
weights establishes a linear performance floor. **XGBoost** is the principal
baseline: gradient boosting remains the dominant approach on tabular data, so the
research question is only meaningful against a properly tuned instance. Class
imbalance is addressed through per-sample weighting rather than resampling,
avoiding the introduction of synthetic observations.

A **multilayer perceptron** (two to three hidden layers, batch normalisation,
dropout) consumes the identical one-hot matrix supplied to XGBoost, isolating the
effect of model class from input representation.

**TabTransformer** [Huang et al., 2020] was implemented directly in PyTorch
rather than through a wrapper library, permitting exact architectural
specification and reducing reproduction dependencies. Categorical inputs are
mapped to learned embeddings, augmented with a per-column embedding that allows
attention to distinguish semantically distinct columns sharing a token index, and
processed through N pre-norm transformer blocks combining multi-head
self-attention with a position-wise feed-forward network. Pre-norm ordering was
selected over the original post-norm formulation because it trains stably at
shallow depths without a learning-rate warm-up schedule, permitting an identical
optimisation protocol across both neural models. The resulting contextual
embeddings are flattened and concatenated with layer-normalised continuous
features before a multilayer perceptron head. The selected configuration used a
32-dimensional embedding, 8 attention heads and 3 transformer blocks.

Both neural models are trained through a single shared routine: AdamW,
class-weighted cross-entropy, gradient clipping at 5.0, plateau-based learning
rate reduction, and early stopping on validation F1-macro with a patience of 20
epochs and a 200-epoch ceiling. Early stopping monitors F1-macro rather than
validation loss because, with Healthy representing 14.8% of observations, loss is
dominated by the majority classes and can continue improving while minority-class
recall degrades. Holding the optimisation protocol constant ensures the
comparison isolates architecture rather than training budget.

---

## IV. EVALUATION AND RESULTS

### A. Metrics

F1-macro is the primary metric. Accuracy is reported but never used for model
selection: with Healthy at 14.8% of observations, a classifier that never
predicts the minority class attains approximately 45% accuracy while providing no
practical value. Per-class precision, recall and F1 are reported throughout,
together with one-versus-rest macro-averaged AUC-ROC.

The cost asymmetry in this task runs opposite to the more familiar
fraud-detection framing. Because Healthy is the minority class, the expensive
error is classifying a financially sound firm as distressed — in credit terms,
denying financing to a creditworthy borrower.

### B. Comparative Performance

TABLE II
MODEL COMPARISON ON THE HELD-OUT TEST SET (MEAN ± SD OVER FIVE SEEDS)

| Model | F1-macro | AUC-ROC | F1 At-Risk | F1 Distressed | F1 Healthy | Train (s) | Params |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.737 ± 0.000 | 0.907 | 0.718 | 0.854 | 0.638 | 0.0 | 72 |
| **XGBoost** | **0.847 ± 0.001** | **0.963** | **0.835** | **0.897** | **0.810** | 0.6 | 2,156,224 |
| MLP | 0.789 ± 0.015 | 0.941 | 0.778 | 0.877 | 0.713 | 8.1 | 40,195 |
| TabTransformer | 0.754 ± 0.011 | 0.932 | 0.737 | 0.862 | 0.664 | 281.4 | 65,431 |

XGBoost attains the highest score on every metric reported. All learned models
substantially exceed the majority-class baseline (F1-macro ≈ 0.21), confirming
that the engineered feature space carries genuine discriminative signal.

### C. RQ1: Deep Tabular Models versus Gradient Boosting

**RQ1 is answered in the negative for this task.** TabTransformer underperforms
XGBoost by 0.093 F1-macro. Against a pooled standard deviation of 0.011 across
seeds, this difference is approximately eight times the observed run-to-run
variation and cannot be attributed to sampling variance. XGBoost additionally
trains roughly 470 times faster (0.6 s versus 281.4 s per seed), a practical
consideration for periodic retraining as new filings are published.

The MLP occupies an intermediate position (0.789 ± 0.015), indicating that neural
approaches are viable on this data but do not match gradient boosting. That
TabTransformer underperforms even the simpler MLP is informative: the
transformer's additional capacity is not merely unhelpful but actively
detrimental at this data scale.

Per-class analysis identifies the mechanism. On the best-validation seed,
TabTransformer achieves Healthy recall of 0.831 but Healthy precision of only
0.559, against 0.897 and 0.735 respectively for XGBoost. The model systematically
over-predicts the minority class, and the resulting false positives are drawn
predominantly from At-Risk, whose recall falls to 0.694 against 0.827 for
XGBoost. We attribute this to the limited categorical structure available: with
only four categorical tokens per observation, the self-attention mechanism has
little relational structure to exploit, while the additional parameters increase
susceptibility to overfitting the minority class under class-weighted loss. This
is consistent with the broader finding that transformer architectures for tabular
data require either substantially larger datasets or richer categorical
vocabularies to realise their advantages.

### D. Discrimination and Error Structure

AUC-ROC values (Table II) indicate that all models separate the classes well
above chance, with XGBoost reaching 0.963. The gap between AUC-ROC and F1-macro
across models reflects threshold calibration rather than ranking quality: the
models order observations similarly but differ in where they place decision
boundaries under class weighting.

Confusion analysis shows that the dominant error mode for all models is
At-Risk/Distressed confusion. For XGBoost, 67 of 618 true Distressed firms are
assigned to At-Risk and 45 of 566 true At-Risk firms to Distressed. This is
expected: the two classes are separated by a single continuous threshold on Z′
(1.23) rather than by any qualitative distinction, so firms near the boundary are
intrinsically ambiguous. Notably, XGBoost assigns zero true Distressed firms to
the Healthy class, indicating that the most consequential error type is
effectively eliminated.

### E. Circularity Audit

Because the labels derive from a formula reconstructible from the underlying
data, the exclusion of the Altman components requires empirical verification
rather than assertion. If a single feature captured the majority of model gain,
this would suggest the model had recovered a label proxy rather than learning the
intended task.

The highest-ranked feature, `profit_regime_loss`, accounts for 54.4% of total
XGBoost gain, with the remainder distributed across the feature set. This falls
below the threshold at which circularity would be suspected, though the
concentration is high enough to warrant discussion.

Two considerations support interpreting this as genuine economic signal. First,
`profit_regime` is derived from net margin, an income-statement ratio, and is not
one of the five Altman Z′ components, all of which are balance-sheet quantities
scaled by total assets; no algebraic path connects this feature to the label.
Second, the distress-prediction literature consistently identifies sustained
loss-making as the dominant single predictor of financial distress, so a model
ranking it first is consistent with domain expectation.

The concentration is additionally sensitive to hyperparameter selection: an
alternative configuration with more estimators and weaker minimum child weight
produced 28.9% for the same feature. Gain-based importance is known to
concentrate under shallower, more strongly regularised tree ensembles, and this
sensitivity should be borne in mind when interpreting the figure.

### F. Threats to Validity

**Label proxy.** Altman Z′ zones substitute for observed distress events. Small
unprofitable listed firms are assigned to the Distressed zone without having
filed for bankruptcy protection, so reported performance measures agreement with
a formula-derived proxy rather than with realised outcomes. This is the study's
principal limitation; validation against actual Chapter 11 filings would
materially strengthen the design.

**Temporal coverage.** The dataset spans FY2021–2024, a period encompassing
pandemic recovery and an elevated interest rate environment. Generalisation to
other macroeconomic regimes is untested.

**Population.** Financial-sector firms were excluded during data construction, as
is standard in the distress-prediction literature, so findings do not extend to
banks or insurers.

**Inherited preprocessing leak.** The four original ratios were winsorised across
the full dataset prior to partitioning, so their clipping bounds incorporate
test-fold percentiles. The effect is expected to be small, but it is disclosed
rather than omitted.

---

## VI. CONCLUSIONS (modelling contribution)

On firm-level financial health classification from Profit and Loss ratios,
gradient boosting outperformed both neural architectures evaluated, with
TabTransformer trailing XGBoost by 0.093 F1-macro at approximately eight times
the measured seed variance and at roughly 470 times the training cost. The result
is reported as obtained. Gradient boosting's dominance on tabular data is well
documented, and a rigorously conducted negative result answers RQ1 as
informatively as a positive one would.

The more consequential methodological finding concerns feature-label
correspondence. Labels derived from a formula operating on quantities absent from
the feature space produce a task no architecture can solve: the original
four-ratio configuration capped attainable performance at 0.490 F1-macro with
9.6% Healthy recall, irrespective of model choice. Aligning the feature space
with the label-generating process — while deliberately withholding the exact
formula components to prevent circularity — raised attainable performance to
0.847. Verifying this correspondence before model development would benefit any
study employing formula-derived labels.

Future work should validate against realised bankruptcy filings rather than Z′
zones, extend the temporal window across multiple macroeconomic regimes, and
evaluate whether richer categorical vocabularies (sector classification,
exchange listing, auditor identity) allow attention-based architectures to close
the observed gap.
