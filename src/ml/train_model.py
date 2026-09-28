from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/sample/transactions_with_fraud.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fraud_random_forest.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("REAL-TIME FRAUD DETECTION - ML TRAINING")
    print("=" * 70)

    print(f"\nLoading dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    fraud_count = int(df["is_fraud"].sum())
    normal_count = len(df) - fraud_count

    print(f"Total transactions : {len(df)}")
    print(f"Total columns      : {len(df.columns)}")
    print(f"Normal transactions: {normal_count}")
    print(f"Fraud transactions : {fraud_count}")
    print(f"Fraud rate          : {fraud_count / len(df) * 100:.2f}%")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    print("\nCreating ML features...")

    df = df.copy()

    # Parse transaction timestamp.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    # Time-based features.
    df["transaction_hour"] = df["timestamp"].dt.hour

    df["transaction_day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    df["night_transaction"] = (
        df["transaction_hour"]
        .between(0, 4)
        .astype(int)
    )

    # Amount-based features.
    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    df["large_transaction"] = (
        df["amount"] >= 5000
    ).astype(int)

    df["very_large_transaction"] = (
        df["amount"] >= 10000
    ).astype(int)

    # Geographic feature.
    df["international_transaction"] = (
        df["country"]
        .fillna("UNKNOWN")
        .ne("US")
        .astype(int)
    )

    print("Feature engineering completed.")

    return df


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(df):
    print("\nSelecting model features...")

    # IMPORTANT:
    #
    # We intentionally DO NOT use:
    #
    # is_fraud
    # fraud_type
    # new_device_flag
    # high_amount_flag
    # geo_anomaly_flag
    # velocity_flag
    #
    # is_fraud is the prediction target.
    # fraud_type reveals the answer.
    #
    # The four *_flag fields were used during synthetic fraud
    # generation, so including them could introduce target leakage.

    numeric_features = [
        "amount",
        "log_amount",
        "latitude",
        "longitude",
        "transaction_hour",
        "transaction_day_of_week",
        "is_weekend",
        "night_transaction",
        "large_transaction",
        "very_large_transaction",
        "international_transaction",
    ]

    categorical_features = [
        "transaction_type",
        "channel",
        "merchant_category",
        "country",
    ]

    feature_columns = (
        numeric_features + categorical_features
    )

    X = df[feature_columns].copy()
    y = df["is_fraud"].astype(int)

    print(f"Features selected    : {len(feature_columns)}")
    print(f"Numeric features     : {len(numeric_features)}")
    print(f"Categorical features : {len(categorical_features)}")

    return (
        X,
        y,
        numeric_features,
        categorical_features,
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")
    print(f"Training fraud  : {int(y_train.sum())}")
    print(f"Testing fraud   : {int(y_test.sum())}")

    return X_train, X_test, y_train, y_test


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(
    numeric_features,
    categorical_features,
):
    print("\nBuilding preprocessing and ML pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    pipeline,
    X_train,
    y_train,
):
    print("\nTraining Random Forest fraud model...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Model training completed successfully.")

    return pipeline


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    pipeline,
    X_test,
    y_test,
):
    print("\nEvaluating model on unseen test data...")

    predictions = pipeline.predict(X_test)

    probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    print()
    print("=" * 70)
    print("FRAUD MODEL EVALUATION")
    print("=" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print()
    print("Confusion Matrix")
    print("-" * 70)

    print(f"True Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("=" * 70)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(pipeline):
    print("\nSaving trained fraud detection model...")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Load the complete labeled fraud dataset.
    df = load_data()

    # Engineer transaction features.
    df = create_features(df)

    # Separate predictors and target.
    (
        X,
        y,
        numeric_features,
        categorical_features,
    ) = select_features(df)

    # Create stratified train/test datasets.
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    # Build preprocessing + Random Forest pipeline.
    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
    )

    # Train model.
    pipeline = train_model(
        pipeline,
        X_train,
        y_train,
    )

    # Evaluate model.
    evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    # Save complete preprocessing + ML pipeline.
    save_model(pipeline)

    print()
    print("=" * 70)
    print("ML TRAINING PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()