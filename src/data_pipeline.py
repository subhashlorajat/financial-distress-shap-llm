# Member 1 (Data Engineer)

# Steps 1-3:
# - Load SimFin financial statements
# - Calculate financial ratios
# - Generate Altman Z' score labels

# Input:
#     SimFin annual statements

# Output:
    # labelled.csv and labelled_full.csv

from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# load raw financial statements
def load_raw():
    inc = pd.read_csv(RAW_DIR / "us-income-annual.csv", sep=";")
    bal = pd.read_csv(RAW_DIR / "us-balance-annual.csv", sep=";")
    return inc, bal


# build features from income and balance sheets
def build_features(inc: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    # select required fields and rename for easier use
    # SimFin stores expenses as negative values
    inc = inc.rename(columns={
        "Ticker": "company_id", "Fiscal Year": "year",
        "Cost of Revenue": "cogs", "Operating Expenses": "opex",
        "Operating Income (Loss)": "ebit", "Net Income": "net_income",
        "Revenue": "revenue",
    })[["company_id", "year", "revenue", "cogs", "opex", "ebit", "net_income"]]

    bal = bal.rename(columns={
        "Ticker": "company_id", "Fiscal Year": "year",
        "Total Current Assets": "curr_assets",
        "Total Current Liabilities": "curr_liab",
        "Retained Earnings": "retained_earnings",
        "Total Assets": "total_assets",
        "Total Liabilities": "total_liab",
        "Total Equity": "book_equity",
    })[["company_id", "year", "curr_assets", "curr_liab",
        "retained_earnings", "total_assets", "total_liab", "book_equity"]]

    df = inc.merge(bal, on=["company_id", "year"], how="inner")

    # convert expenses back to positive values
    df["cogs"] = df["cogs"].abs()
    df["opex"] = df["opex"].abs()

    # remove incomplete records
    df = df[df["revenue"].notna() & (df["revenue"] > 0)]
    df = df[df["cogs"].notna()]
    df = df.dropna(subset=["curr_assets", "curr_liab", "retained_earnings",
                           "total_assets", "total_liab", "book_equity",
                           "net_income", "ebit", "opex"])
    df = df[df["total_assets"] > 0]

    # calculate margins
    df["gross_margin"] = (df["revenue"] - df["cogs"]) / df["revenue"]
    df["net_margin"] = df["net_income"] / df["revenue"]

    # calculate yearly growth values
    df = df.sort_values(["company_id", "year"])
    df["total_cost"] = df["cogs"] + df["opex"]
    prev = df.groupby("company_id")[["revenue", "total_cost", "year"]].shift(1)
    consecutive = (df["year"] - prev["year"]) == 1

    df["revenue_growth"] = np.where(
        consecutive & (prev["revenue"] > 0),
        (df["revenue"] - prev["revenue"]) / prev["revenue"], np.nan)

    df["cost_growth"] = np.where(
        consecutive & (prev["total_cost"] > 0),
        (df["total_cost"] - prev["total_cost"]) / prev["total_cost"], np.nan)

    # first year has no previous data
    df = df.dropna(subset=["revenue_growth", "cost_growth"])
    return df


# calculate Altman score and assign classes
def add_altman_labels(df: pd.DataFrame) -> pd.DataFrame:
    ta = df["total_assets"]
    x1 = (df["curr_assets"] - df["curr_liab"]) / ta
    x2 = df["retained_earnings"] / ta
    x3 = df["ebit"] / ta
    x4 = df["book_equity"] / df["total_liab"].replace(0, np.nan)
    x5 = df["revenue"] / ta

    df["z_score"] = (0.717 * x1 + 0.847 * x2 + 3.107 * x3
                     + 0.420 * x4 + 0.998 * x5)

    df = df.dropna(subset=["z_score"])

    df["label"] = pd.cut(
        df["z_score"],
        bins=[-np.inf, 1.23, 2.90, np.inf],
        labels=["Distressed", "At-Risk", "Healthy"],
    )
    return df


def main():
    inc, bal = load_raw()
    n_raw = len(inc)

    df = build_features(inc, bal)
    df = add_altman_labels(df)

    # limit extreme values before model training
    for c in ["gross_margin", "net_margin", "revenue_growth", "cost_growth"]:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)

    core_cols = ["company_id", "year", "gross_margin", "net_margin",
                 "revenue_growth", "cost_growth", "label"]

    df[core_cols].to_csv(OUT_DIR / "labelled.csv", index=False)
    df.to_csv(OUT_DIR / "labelled_full.csv", index=False)

    print(f"Raw income-statement rows : {n_raw}")
    print(f"Final labelled rows       : {len(df)}")
    print(f"Companies                 : {df['company_id'].nunique()}")
    print(f"Years                     : {df['year'].min()}–{df['year'].max()}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print("\nSaved -> data/labelled.csv and data/labelled_full.csv")


if __name__ == "__main__":
    main()