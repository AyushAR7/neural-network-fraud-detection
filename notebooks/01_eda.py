"""
Step 1: Data Loading + EDA + Stratified Split
Neural Network from Scratch + Fraud Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# ---- Load data ----
df = pd.read_csv("data/fraud_detection.csv")  # adjust path if needed

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print("\nDTYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isnull().sum())
print("\nDUPLICATES:", df.duplicated().sum())
print("\nDESCRIBE (numeric):\n", df.describe().T)

print("\n" + "=" * 60)
print("CLASS BALANCE (is_fraud)")
print("=" * 60)
print(df["is_fraud"].value_counts())
print(df["is_fraud"].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("MERCHANT CATEGORY VALUE COUNTS")
print("=" * 60)
print(df["merchant_category"].value_counts())

# ---- Feature means split by class (great EDA signal for imbalanced data) ----
numeric_cols = ["amount", "transaction_hour", "foreign_transaction", "location_mismatch",
                 "device_trust_score", "velocity_last_24h", "cardholder_age"]
print("\n" + "=" * 60)
print("FEATURE MEANS BY CLASS (fraud vs not fraud)")
print("=" * 60)
print(df.groupby("is_fraud")[numeric_cols].mean().T)

# ---- Correlation with target ----
print("\n" + "=" * 60)
print("CORRELATION WITH TARGET (is_fraud)")
print("=" * 60)
print(df[numeric_cols + ["is_fraud"]].corr()["is_fraud"].sort_values(ascending=False))

# ---- Stratified train/test split (critical given only 151 fraud cases) ----
X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("STRATIFIED SPLIT CHECK")
print("=" * 60)
print("Train fraud ratio:", y_train.mean())
print("Test fraud ratio:", y_test.mean())
print("Train size:", X_train.shape, " Test size:", X_test.shape)

# Save splits for the next steps
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("\nSaved stratified train/test splits.")

# ---- Plots ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

df["is_fraud"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["#1F4E79", "#C0392B"])
axes[0, 0].set_title("Class Balance (0 = Legit, 1 = Fraud)")
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(["Legit", "Fraud"], rotation=0)

sns.boxplot(data=df, x="is_fraud", y="amount", ax=axes[0, 1], hue="is_fraud",
            palette=["#1F4E79", "#C0392B"], legend=False)
axes[0, 1].set_title("Transaction Amount by Class")
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(["Legit", "Fraud"])

sns.boxplot(data=df, x="is_fraud", y="device_trust_score", ax=axes[1, 0], hue="is_fraud",
            palette=["#1F4E79", "#C0392B"], legend=False)
axes[1, 0].set_title("Device Trust Score by Class")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["Legit", "Fraud"])

sns.boxplot(data=df, x="is_fraud", y="velocity_last_24h", ax=axes[1, 1], hue="is_fraud",
            palette=["#1F4E79", "#C0392B"], legend=False)
axes[1, 1].set_title("Transaction Velocity (24h) by Class")
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(["Legit", "Fraud"])

plt.tight_layout()
plt.savefig("eda_overview.png")
print("\nSaved eda_overview.png")

# ---- Correlation heatmap ----
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols + ["is_fraud"]].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("Saved correlation_heatmap.png")