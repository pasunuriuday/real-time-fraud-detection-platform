import os

import pandas as pd
import psycopg


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("FRAUD_DB_HOST", "localhost")
DB_PORT = os.getenv("FRAUD_DB_PORT", "5432")
DB_NAME = os.getenv("FRAUD_DB_NAME", "fraud_detection")
DB_USER = os.getenv("FRAUD_DB_USER", "saleor")
DB_PASSWORD = os.getenv("FRAUD_DB_PASSWORD", "saleor")


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

def load_predictions():

    query = """
        SELECT
            transaction_id,
            fraud_probability,
            actual_fraud
        FROM fraud_predictions
        WHERE fraud_probability IS NOT NULL
          AND actual_fraud IS NOT NULL;
    """

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    ) as connection:

        return pd.read_sql_query(
            query,
            connection,
        )


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(df, threshold):

    predicted = (
        df["fraud_probability"] >= threshold
    ).astype(int)

    actual = df["actual_fraud"].astype(int)

    tp = int(
        ((predicted == 1) & (actual == 1)).sum()
    )

    fp = int(
        ((predicted == 1) & (actual == 0)).sum()
    )

    fn = int(
        ((predicted == 0) & (actual == 1)).sum()
    )

    tn = int(
        ((predicted == 0) & (actual == 0)).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (tp + tn)
        / len(df)
        if len(df) > 0
        else 0
    )

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("FRAUD DETECTION - THRESHOLD ANALYSIS")
    print("=" * 90)

    print()
    print("Loading predictions from PostgreSQL...")

    df = load_predictions()

    df["fraud_probability"] = pd.to_numeric(
        df["fraud_probability"],
        errors="coerce",
    )

    df["actual_fraud"] = pd.to_numeric(
        df["actual_fraud"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "fraud_probability",
            "actual_fraud",
        ]
    )

    print(
        f"Transactions evaluated : {len(df)}"
    )

    print(
        f"Actual fraud cases     : "
        f"{int(df['actual_fraud'].sum())}"
    )

    print()

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
    ]

    results = []

    for threshold in thresholds:

        results.append(
            evaluate_threshold(
                df,
                threshold,
            )
        )

    results_df = pd.DataFrame(
        results
    )

    results_df["precision"] = (
        results_df["precision"] * 100
    ).round(2)

    results_df["recall"] = (
        results_df["recall"] * 100
    ).round(2)

    results_df["f1"] = (
        results_df["f1"] * 100
    ).round(2)

    results_df["accuracy"] = (
        results_df["accuracy"] * 100
    ).round(2)

    print("=" * 90)
    print("THRESHOLD COMPARISON")
    print("=" * 90)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("=" * 90)
    print("BEST F1 THRESHOLD")
    print("=" * 90)

    print(
        f"Threshold : {best_row['threshold']:.2f}"
    )

    print(
        f"Precision : {best_row['precision']:.2f}%"
    )

    print(
        f"Recall    : {best_row['recall']:.2f}%"
    )

    print(
        f"F1 Score  : {best_row['f1']:.2f}%"
    )

    print(
        f"False Positives : {int(best_row['FP'])}"
    )

    print(
        f"False Negatives : {int(best_row['FN'])}"
    )

    print()

    # --------------------------------------------------------
    # Actual fraud probabilities
    # --------------------------------------------------------

    print("=" * 90)
    print("ACTUAL FRAUD TRANSACTIONS")
    print("=" * 90)

    fraud_cases = (
        df[
            df["actual_fraud"] == 1
        ]
        [
            [
                "transaction_id",
                "fraud_probability",
            ]
        ]
        .sort_values(
            "fraud_probability",
            ascending=False,
        )
    )

    fraud_cases[
        "fraud_probability"
    ] = (
        fraud_cases[
            "fraud_probability"
        ]
        * 100
    ).round(2)

    print()

    print(
        fraud_cases.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("THRESHOLD ANALYSIS COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()