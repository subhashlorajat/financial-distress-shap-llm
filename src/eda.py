"""Member 1 - Exploratory Data Analysis plots."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE / "data" / "labelled_full.csv")
OUT = BASE / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
order = ["Healthy", "At-Risk", "Distressed"]
palette = {"Healthy": "#2a9d8f", "At-Risk": "#e9c46a", "Distressed": "#e76f51"}

# class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["label"].value_counts().reindex(order)
ax.bar(counts.index, counts.values, color=[palette[c] for c in order])
for i, v in enumerate(counts.values):
    ax.text(i, v + 40, f"{v}\n({v/len(df):.0%})", ha="center", fontsize=9)
ax.set_ylabel("Company-years")
ax.set_title("Class distribution of Altman Z' labels (2021–2025)")
fig.tight_layout()
fig.savefig(OUT / "01_class_distribution.png", dpi=200)

# ratio distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col, title in zip(
    axes.ravel(),
    ["gross_margin", "net_margin", "revenue_growth", "cost_growth"],
    ["Gross Margin", "Net Margin", "Revenue Growth", "Cost Growth"],
):
    series = df[col].clip(-2, 2) if col == "net_margin" else df[col]
    for lab in order:
        sns.kdeplot(series[df["label"] == lab], ax=ax,
                    label=lab, color=palette[lab], fill=True, alpha=0.25)
    ax.set_title(title + (" (display clipped to ±2)" if col == "net_margin" else ""))
    ax.set_xlabel("")
axes[0, 0].legend()
fig.suptitle("Distribution of the four model ratios by class", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "02_ratio_distributions.png", dpi=200, bbox_inches="tight")

# z-score distribution
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["z_score"].clip(-10, 15), bins=80, ax=ax, color="#457b9d")
ax.axvline(1.23, color="#e76f51", ls="--", label="Z' = 1.23 (distress)")
ax.axvline(2.90, color="#2a9d8f", ls="--", label="Z' = 2.90 (safe)")
ax.set_xlabel("Altman Z'-score (clipped to [-10, 15] for display)")
ax.set_title("Z'-score distribution with zone thresholds")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "03_zscore_distribution.png", dpi=200)

# yearly class proportions
fig, ax = plt.subplots(figsize=(7, 4))
share = (df.groupby(["year", "label"], observed=True).size()
           .unstack()[order].apply(lambda r: r / r.sum(), axis=1))
share.plot(kind="bar", stacked=True, ax=ax,
           color=[palette[c] for c in order], width=0.75)
ax.set_ylabel("Share of company-years")
ax.set_title("Label composition by fiscal year")
ax.legend(title="", loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "04_class_by_year.png", dpi=200)

# check ratio outliers
fig, ax = plt.subplots(figsize=(8, 4))
melted = df.melt(id_vars="label",
                 value_vars=["gross_margin", "net_margin",
                             "revenue_growth", "cost_growth"],
                 var_name="ratio", value_name="value")
sns.boxplot(data=melted, x="ratio", y="value", hue="label",
            hue_order=order, palette=palette, ax=ax, fliersize=1)
ax.set_title("Ratio spread and outliers by class (after 1%/99% winsorisation)")
ax.set_xlabel("")
fig.tight_layout()
fig.savefig(OUT / "05_ratio_boxplots.png", dpi=200)

print("Saved 5 plots to results/plots/")
print(df.groupby("label", observed=True)[
    ["gross_margin", "net_margin", "revenue_growth", "cost_growth"]
].median().round(3).to_string())