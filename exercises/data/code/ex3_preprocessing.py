"""
Exercise 3 - Preparing Real-World Data for a Neural Network
Spaceship Titanic (train.csv). Generates Figure 6 and the numbers used in
the report / results summary.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

RNG_SEED = 42
FIG_DIR = "../figures"

df = pd.read_csv("data/train.csv")

# ----- A: Get to know the data --------------------------------------------------
target_balance = df["Transported"].value_counts(normalize=True)

numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_features = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

missing_table = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
})
missing_table = missing_table[missing_table["missing_count"] > 0].sort_values(
    "missing_count", ascending=False
)

spending_stats = df[spending_cols].agg(["mean", "median", "max"]).T

# ----- B: Split before you transform --------------------------------------------
y = df["Transported"]
X = df.drop(columns=["Transported"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG_SEED
)

# mean/median of FoodCourt on the TRAINING set, before any transform (for results summary)
foodcourt_mean_before = X_train["FoodCourt"].mean()
foodcourt_median_before = X_train["FoodCourt"].median()

# ----- C: Preprocess --------------------------------------------------------------
def engineer(frame):
    frame = frame.copy()
    frame["TotalSpend"] = frame[spending_cols].sum(axis=1, skipna=True)
    frame = frame.drop(columns=["Cabin", "Name", "PassengerId"])
    return frame

X_train_fe = engineer(X_train)
X_test_fe = engineer(X_test)

num_cols_final = numerical_features + ["TotalSpend"]
log_cols = spending_cols + ["TotalSpend"]  # heavy-tailed columns get log1p

# Save a copy of one heavy-tailed column BEFORE the log transform, for Figure 6
foodcourt_train_before = X_train_fe["FoodCourt"].copy()

# Impute (fit on train only)
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_fe[num_cols_final] = num_imputer.fit_transform(X_train_fe[num_cols_final])
X_test_fe[num_cols_final] = num_imputer.transform(X_test_fe[num_cols_final])

X_train_fe[categorical_features] = cat_imputer.fit_transform(X_train_fe[categorical_features])
X_test_fe[categorical_features] = cat_imputer.transform(X_test_fe[categorical_features])

# log1p on the heavy-tailed spending columns (incl. engineered TotalSpend)
for c in log_cols:
    X_train_fe[c] = np.log1p(X_train_fe[c])
    X_test_fe[c] = np.log1p(X_test_fe[c])

foodcourt_train_after = X_train_fe["FoodCourt"].copy()

# One-hot encode categoricals, fit on train only; unseen test categories -> all-zero row
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_train = ohe.fit_transform(X_train_fe[categorical_features])
ohe_test = ohe.transform(X_test_fe[categorical_features])
ohe_cols = ohe.get_feature_names_out(categorical_features)

# Scale numeric columns (standardization), fit on train only
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train_fe[num_cols_final])
scaled_test = scaler.transform(X_test_fe[num_cols_final])

X_train_final = np.hstack([scaled_train, ohe_train])
X_test_final = np.hstack([scaled_test, ohe_test])
final_columns = num_cols_final + list(ohe_cols)

train_final_df = pd.DataFrame(X_train_final, columns=final_columns)
test_final_df = pd.DataFrame(X_test_final, columns=final_columns)

# ----- D: Verify and visualize ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(foodcourt_train_before, bins=40, color="#1f77b4")
axes[0].set_title("FoodCourt — before (raw, training set)")
axes[0].set_xlabel("FoodCourt ($)")
axes[0].set_ylabel("count")

axes[1].hist(foodcourt_train_after, bins=40, color="#d62728")
axes[1].set_title("FoodCourt — after log1p (training set)")
axes[1].set_xlabel("log1p(FoodCourt)")
axes[1].set_ylabel("count")

fig.suptitle("Figure 6 — Effect of the log1p transform on a heavy-tailed feature")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig6_foodcourt_before_after.png", dpi=150)
plt.close(fig)

n_nan_train = int(train_final_df.isnull().sum().sum())
n_nan_test = int(test_final_df.isnull().sum().sum())
final_shape_train = train_final_df.shape
train_min = float(scaled_train.min())
train_max = float(scaled_train.max())
test_min = float(scaled_test.min())
test_max = float(scaled_test.max())

results = {
    "target_balance": target_balance.to_dict(),
    "missing_table": missing_table.to_dict(orient="index"),
    "spending_stats": spending_stats.round(2).to_dict(orient="index"),
    "foodcourt_mean_before": float(foodcourt_mean_before),
    "foodcourt_median_before": float(foodcourt_median_before),
    "n_nan_train_final": n_nan_train,
    "n_nan_test_final": n_nan_test,
    "final_shape_train": list(final_shape_train),
    "final_shape_test": list(test_final_df.shape),
    "scaled_train_min": train_min,
    "scaled_train_max": train_max,
    "scaled_test_min": test_min,
    "scaled_test_max": test_max,
}
with open("ex3_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
