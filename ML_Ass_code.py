# basmala 1220184, yasmin 1220848

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

DATA_PATH = r"C:/Users/Asus/OneDrive/Documents/folder_python/ML_ass1/customer_data.csv"
OUTPUT_DIR = "assignment_outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def tidy_axes(title=None, xlabel=None, ylabel=None):
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.tight_layout()

MAX_CAT_BARS = 30

def is_small_categorical(series, max_unique=MAX_CAT_BARS):
    try:
        return series.nunique(dropna=False) <= max_unique
    except Exception:
        return False


# 1) DATA LOADING & INITIAL INSPECTION
print("\n[1] Data Loading & Initial Inspection")

df = pd.read_csv(DATA_PATH)
print("\n.head() function output: ")
print(df.head())
print("\n.info() function output: ")
print(df.info())
print("\n.describe() function output (numeric): ")
print(df.describe(include=[np.number]).T)

ID_COL = "CustomerID" if "CustomerID" in df.columns else None
TARGET = "ChurnStatus" if "ChurnStatus" in df.columns else None

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in ["Gender", "ProductType"]:
    if c in df.columns and pd.api.types.is_integer_dtype(df[c]):
        df[c] = df[c].astype("category")
        if c in num_cols:
            num_cols.remove(c)
        if c not in cat_cols:
            cat_cols.append(c)


# 2) HANDLING MISSING DATA
print("\n[2] Handling Missing Data")

print("\nMissing values (count): ")
print(df.isnull().sum())
print("\nMissing values (percent): ")
print((df.isnull().mean() * 100).round(2))

protected = [c for c in [ID_COL, TARGET] if c is not None]
num_impute = [c for c in num_cols if c not in protected]
cat_impute = [c for c in cat_cols if c not in protected]

for c in num_impute:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_impute:
    if df[c].isnull().any():
        mode_val = df[c].mode(dropna=True)
        mode_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df[c] = df[c].fillna(mode_val)

print("\nRemaining missing after imputation: ")
print(df.isnull().sum())

os.makedirs(OUTPUT_DIR, exist_ok=True)
clean_path = os.path.join(OUTPUT_DIR, "customer_data_cleaned.csv")
df.to_csv(clean_path, index=False)
print(f"Saved cleaned data to: {clean_path}")

# 3) HANDLING OUTLIERS (Z-score count + BINNING median smoothing)
print("\n[3] Handling Outliers")

def zscore_flags(frame, cols, thresh=3.0):
    cols = [c for c in cols if c in frame.columns]
    z = (frame[cols] - frame[cols].mean()) / frame[cols].std(ddof=0)
    return (z.abs() > thresh)

num_for_outliers = [c for c in num_cols if c not in protected and pd.api.types.is_numeric_dtype(df[c])]

if len(num_for_outliers) > 0:
    z_flags = zscore_flags(df, num_for_outliers, 3.0)
    outlier_counts = z_flags.sum().sort_values(ascending=False)
    print("\nOutlier counts by feature (|z|>3): ")
    print(outlier_counts)

    BINS = 10
    print(f"\nApplying binning-based smoothing to numeric features (q={BINS}, replace with bin medians).")
    for c in num_for_outliers:
        if df[c].nunique(dropna=True) <= 1:
            continue
        before_std = df[c].std(ddof=0)
        try:
            bin_codes = pd.qcut(df[c], q=BINS, duplicates='drop')
        except ValueError:
            bin_count = min(BINS, max(2, df[c].nunique()))
            bin_codes = pd.cut(df[c], bins=bin_count, include_lowest=True)

        bin_medians = df.groupby(bin_codes, observed=True)[c].median()
        df[c] = bin_codes.map(bin_medians)

        df[c] = pd.to_numeric(df[c], errors="coerce")

        after_std = df[c].std(ddof=0)
        print(f"{c}: std before -> after = {before_std:.3f} -> {after_std:.3f}")

post_outlier_path = os.path.join(OUTPUT_DIR, "customer_data_post_outliers.csv")
df.to_csv(post_outlier_path, index=False)
print(f"Saved post-outlier data to: {post_outlier_path}")

# 4) FEATURE SCALING (Min–Max + Standardization)
print("\n[4] Feature Scaling")

from pandas.api.types import is_numeric_dtype

num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
scale_cols = [c for c in num_cols if c not in [col for col in [ID_COL, TARGET] if col is not None]]

df_std = df.copy()
df_mm  = df.copy()

for c in scale_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if pd.notna(sigma) and sigma != 0:
        df_std[c] = (s - mu) / sigma

for c in scale_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    mn = s.min()
    mx = s.max()
    rng = mx - mn
    if pd.notna(rng) and rng != 0:
        df_mm[c] = (s - mn) / rng

std_path = os.path.join(OUTPUT_DIR, "customer_data_standardized.csv")
mm_path  = os.path.join(OUTPUT_DIR, "customer_data_minmax.csv")
df_std.to_csv(std_path, index=False)
df_mm.to_csv(mm_path, index=False)
print(f"Saved standardized data to: {std_path}")
print(f"Saved min–max data to: {mm_path}")

# 5) EXPLORATORY DATA ANALYSIS (EDA)
print("\n[5] EDA + Visualizations (plots saved to figures file)")

# Univariate: Histograms 
for c in [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]:
    plt.figure()
    df[c].hist(bins=30)
    tidy_axes(title=f"Histogram — {c}", xlabel=c, ylabel="Count")
    plt.savefig(os.path.join(FIG_DIR, f"hist_{c}.png"), dpi=150)
    plt.close()

# Univariate: Box plots 
for c in [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]:
    plt.figure()
    df.boxplot(column=c)
    tidy_axes(title=f"Box plot — {c}", ylabel=c)
    plt.savefig(os.path.join(FIG_DIR, f"box_{c}.png"), dpi=150)
    plt.close()

# Univariate: Bar plots 
for c in [c for c in cat_cols if c != TARGET]:
    if not is_small_categorical(df[c]):
        continue
    plt.figure()
    df[c].value_counts(dropna=False).sort_values(ascending=False).plot(kind="bar")
    tidy_axes(title=f"Bar plot — {c}", xlabel=c, ylabel="Count")
    plt.savefig(os.path.join(FIG_DIR, f"bar_{c}.png"), dpi=150)
    plt.close()

# Bivariate: Numeric vs Target 
if TARGET and TARGET in df.columns:
    for c in [c for c in df.select_dtypes(include=[np.number]).columns if c not in [ID_COL, TARGET]]:
        plt.figure()
        jitter = (np.random.rand(len(df)) - 0.5) * 0.05
        plt.scatter(df[c], df[TARGET] + jitter, s=8)
        tidy_axes(title=f"{c} vs {TARGET}", xlabel=c, ylabel=TARGET)
        plt.savefig(os.path.join(FIG_DIR, f"scatter_{c}vs{TARGET}.png"), dpi=150)
        plt.close()

# Bivariate: Categorical vs Target 
if TARGET and TARGET in df.columns:
    for c in [c for c in cat_cols if c != TARGET]:
        if df[c].nunique(dropna=False) > 0 and is_small_categorical(df[c], MAX_CAT_BARS):
            plt.figure()
            grp = df.groupby(c, observed=True)[TARGET].mean().sort_values(ascending=False)
            grp.plot(kind="bar")
            tidy_axes(title=f"Churn rate by {c}", xlabel=c, ylabel="Mean Churn")
            plt.savefig(os.path.join(FIG_DIR, f"churnrate_by_{c}.png"), dpi=150)
            plt.close()

# Correlation matrix 
num_for_corr = [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]
if len(num_for_corr) > 1:
    corr = df[num_for_corr].corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', aspect='auto')
    plt.xticks(range(len(num_for_corr)), num_for_corr, rotation=45, ha='right')
    plt.yticks(range(len(num_for_corr)), num_for_corr)
    plt.colorbar()
    tidy_axes(title="Correlation Matrix (numerical features)")
    plt.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"), dpi=150)
    plt.close()

    if TARGET in num_for_corr:
        print("\nCorrelations with target: ")
        print(corr[TARGET].sort_values(ascending=False))

# 6) SUMMARY
print("\n[6] Visualizations saved to:", FIG_DIR)

print("\nQuick Summary:")
if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
    print("Overall churn rate:", round(df[TARGET].mean(), 4))
if "SupportCalls" in df.columns:
    print("SupportCalls describe:", df["SupportCalls"].describe().to_dict())
if "ProductType" in df.columns:
    vc = df["ProductType"]
    if isinstance(vc.dtype, CategoricalDtype):
        vc = vc.astype("int64")
    print("ProductType value counts (top):", vc.value_counts().head(5).to_dict())


