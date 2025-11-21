import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# ------------------------------------------------------
# 1. Create New Features
# ------------------------------------------------------
def create_new_features(df):
    df["transaction_value"] = df["Quantity"] * df["Price"]
    df["is_high_value"] = (df["transaction_value"] > df["transaction_value"].median()).astype(int)
    df["price_per_unit"] = df["Price"] / (df["Quantity"] + 1)

    print("✔ Fitur baru dibuat:")
    print("- transaction_value")
    print("- is_high_value")
    print("- price_per_unit")

    return df


# ------------------------------------------------------
# 2. Correlation Table (NO PLOT)
# ------------------------------------------------------
def print_correlation(df):

    df_numeric = df.select_dtypes(include=["int64", "float64"])

    if df_numeric.shape[1] < 2:
        print("⚠ Tidak cukup fitur numerik untuk menghitung korelasi")
        return

    corr = df_numeric.corr()
    print("\n=== Correlation Matrix ===")
    print(corr)


# ------------------------------------------------------
# 3. Scaling + Encoding + PCA
# ------------------------------------------------------
def apply_transformations(df, target_col):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    pca = PCA(n_components=2)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("pca", pca)
    ])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_transformed = pipeline.fit_transform(X)

    print("\n=== PCA Explained Variance ===")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")

    return X_transformed, y, preprocessor


# ------------------------------------------------------
# 4. Feature Importance + Selection
# ------------------------------------------------------
def feature_selection(df, target_col):

    df_model = df.select_dtypes(include=["int64", "float64"]).copy()

    if target_col not in df_model.columns:
        print("❌ ERROR: target_dummy tidak numerik")
        return []

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    print("\n=== Feature Importance ===")
    for name, score in zip(X.columns, model.feature_importances_):
        print(f"{name:20s}: {score:.4f}")

    selector = SelectFromModel(model, prefit=True)
    selected = X.columns[selector.get_support()].tolist()

    print("✔ Fitur terpilih:", selected)

    return selected


# ------------------------------------------------------
# 5. MAIN WRAPPER
# ------------------------------------------------------
def run_feature_engineering(df):

    print("\n===== 5. FEATURE ENGINEERING =====")

    df = create_new_features(df)

    df["target_dummy"] = (df["transaction_value"] > df["transaction_value"].median()).astype(int)
    print("\n✔ Target dummy dibuat")

    print_correlation(df)

    X_transformed, y, preprocessor = apply_transformations(df, "target_dummy")

    selected_features = feature_selection(df, "target_dummy")

    print("\n🟢 Feature Engineering selesai.")
    print("==========================================\n")

    return df, X_transformed, y
