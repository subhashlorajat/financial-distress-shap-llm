# Data Quality Section (Member 1 draft, ~500 words)

## Data Sources, Preparation and Quality

Our dataset is built from SimFin's bulk US annual financial-statement files
(income statement and balance sheet), which are machine-parsed from XBRL data
in 10-K filings lodged with SEC EDGAR. Using a pre-parsed distribution rather
than scraping EDGAR directly gives us the same underlying regulatory data
while avoiding the well-documented inconsistencies of raw XBRL tagging across
filers. The raw extract contains 16,974 company-year observations for 4,390
US public companies covering fiscal years 2020–2025.

The two statement files were merged on ticker and fiscal year. SimFin stores
cost items with a negative sign; these were converted to positive magnitudes
before ratio computation. Four cleaning rules were then applied. First, rows
with missing or non-positive revenue were removed, since all four model
ratios are undefined without a positive revenue denominator. Second, rows
lacking Cost of Revenue (22.1% of the raw file) were excluded; inspection
showed these are almost entirely financial institutions, which do not report
COGS and which the Altman framework explicitly does not cover — their
exclusion is standard practice in the financial-distress literature. Third,
observations missing any balance-sheet item required by the Z-score were
dropped. Fourth, because Revenue Growth and Cost Growth require the same
company's previous consecutive fiscal year, each company's first observed
year cannot be labelled and was removed. The final dataset contains 9,284
company-years for 2,942 companies over fiscal years 2021–2025 — well above
the project's 500–1,000 company target.

Labels were generated with the Altman Z′-Score, the private-firm variant of
the original 1968 model (Altman, 2000). We use Z′ rather than the original
formulation because our dataset contains no market prices, and Z′ replaces
the market value of equity in the leverage term with book equity, with
re-estimated coefficients and zone boundaries (Healthy: Z′ > 2.90; At-Risk:
1.23–2.90; Distressed: Z′ < 1.23). EBIT was proxied by operating income. The
resulting class distribution is imbalanced: 44.6% Distressed, 40.5% At-Risk,
and 14.8% Healthy, which motivates our use of macro-averaged F1 rather than
accuracy in evaluation.

Exploratory analysis confirms the labels behave economically sensibly: the
median net margin of Distressed firms is negative (−9.9%) versus +8.5% for
Healthy firms. A notable nuance is that Distressed firms show the highest
median gross margin (51%), driven by small biotechnology and software firms
whose gross margins are high but whose operating expenses exceed revenue —
evidence that no single ratio suffices and a multivariate classifier is
warranted. To limit the influence of extreme micro-cap outliers, the four
input ratios were winsorised at the 1st and 99th percentiles.

Three limitations remain. The Z′-score is a proxy label rather than observed
bankruptcy, so the Distressed class includes chronically unprofitable but
surviving firms; cross-checking against Chapter 11 filings is left as a
robustness step. The 2020–2025 window includes the COVID-19 shock, which may
inflate distress signals in early years. Finally, survivorship effects in
the data vendor's coverage may under-represent firms delisted before 2020.

## References to add
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the
  prediction of corporate bankruptcy. Journal of Finance, 23(4).
- Altman, E. I. (2000). Predicting financial distress of companies:
  revisiting the Z-Score and ZETA models. Working paper, NYU Stern.
