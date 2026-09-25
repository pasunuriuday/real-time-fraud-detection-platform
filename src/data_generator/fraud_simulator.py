import random
import uuid

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Percentage of transactions that will become fraud.
# 0.03 = approximately 3%
FRAUD_RATE = 0.03

FRAUD_TYPES = [
    "ACCOUNT_TAKEOVER",
    "HIGH_VALUE_FRAUD",
    "GEOLOCATION_ANOMALY",
    "VELOCITY_FRAUD",
]


# ---------------------------------------------------------
# ACCOUNT TAKEOVER
# ---------------------------------------------------------

def inject_account_takeover(df, index):
    """
    Simulate account takeover.

    Typical indicators:
    - New/unrecognized device
    - New IP address
    - High transaction amount
    """

    df.at[index, "is_fraud"] = 1
    df.at[index, "fraud_type"] = "ACCOUNT_TAKEOVER"

    df.at[index, "device_id"] = (
        f"SUSPICIOUS-DEV-{uuid.uuid4().hex[:8].upper()}"
    )

    df.at[index, "ip_address"] = (
        f"185.{random.randint(1, 254)}."
        f"{random.randint(1, 254)}."
        f"{random.randint(1, 254)}"
    )

    df.at[index, "amount"] = round(
        random.uniform(2500, 8000),
        2
    )

    df.at[index, "new_device_flag"] = 1
    df.at[index, "high_amount_flag"] = 1
    df.at[index, "geo_anomaly_flag"] = 0
    df.at[index, "velocity_flag"] = 0


# ---------------------------------------------------------
# HIGH VALUE FRAUD
# ---------------------------------------------------------

def inject_high_value_fraud(df, index):
    """
    Simulate an unusually large transaction.
    """

    df.at[index, "is_fraud"] = 1
    df.at[index, "fraud_type"] = "HIGH_VALUE_FRAUD"

    df.at[index, "amount"] = round(
        random.uniform(5000, 15000),
        2
    )

    df.at[index, "high_amount_flag"] = 1
    df.at[index, "new_device_flag"] = 0
    df.at[index, "geo_anomaly_flag"] = 0
    df.at[index, "velocity_flag"] = 0


# ---------------------------------------------------------
# GEOLOCATION ANOMALY
# ---------------------------------------------------------

def inject_geolocation_anomaly(df, index):
    """
    Simulate a transaction occurring from
    an unusual geographic location.
    """

    suspicious_locations = [
        {
            "city": "London",
            "state": "N/A",
            "country": "GB",
            "latitude": 51.5074,
            "longitude": -0.1278,
        },
        {
            "city": "Lagos",
            "state": "N/A",
            "country": "NG",
            "latitude": 6.5244,
            "longitude": 3.3792,
        },
        {
            "city": "Bucharest",
            "state": "N/A",
            "country": "RO",
            "latitude": 44.4268,
            "longitude": 26.1025,
        },
        {
            "city": "Bangkok",
            "state": "N/A",
            "country": "TH",
            "latitude": 13.7563,
            "longitude": 100.5018,
        },
    ]

    location = random.choice(
        suspicious_locations
    )

    df.at[index, "is_fraud"] = 1
    df.at[index, "fraud_type"] = (
        "GEOLOCATION_ANOMALY"
    )

    df.at[index, "city"] = location["city"]
    df.at[index, "state"] = location["state"]
    df.at[index, "country"] = location["country"]
    df.at[index, "latitude"] = location["latitude"]
    df.at[index, "longitude"] = location["longitude"]

    df.at[index, "geo_anomaly_flag"] = 1
    df.at[index, "new_device_flag"] = 0
    df.at[index, "high_amount_flag"] = 0
    df.at[index, "velocity_flag"] = 0


# ---------------------------------------------------------
# VELOCITY FRAUD
# ---------------------------------------------------------

def inject_velocity_fraud(df, index):
    """
    Mark a transaction as part of a suspicious
    high-frequency transaction pattern.

    A more advanced streaming implementation will
    calculate actual rolling transaction velocity.
    """

    df.at[index, "is_fraud"] = 1
    df.at[index, "fraud_type"] = "VELOCITY_FRAUD"

    df.at[index, "velocity_flag"] = 1
    df.at[index, "new_device_flag"] = 0
    df.at[index, "high_amount_flag"] = 0
    df.at[index, "geo_anomaly_flag"] = 0


# ---------------------------------------------------------
# INITIALIZE RISK FLAGS
# ---------------------------------------------------------

def initialize_risk_flags(df):
    """
    Add fraud/risk indicator columns.
    """

    df["new_device_flag"] = 0
    df["high_amount_flag"] = 0
    df["geo_anomaly_flag"] = 0
    df["velocity_flag"] = 0

    return df


# ---------------------------------------------------------
# INJECT FRAUD
# ---------------------------------------------------------

def inject_fraud(df, fraud_rate=FRAUD_RATE):
    """
    Select a percentage of transactions and
    inject controlled fraud patterns.
    """

    df = initialize_risk_flags(
        df.copy()
    )

    number_of_fraud_transactions = int(
        len(df) * fraud_rate
    )

    fraud_indices = random.sample(
        list(df.index),
        number_of_fraud_transactions
    )

    for index in fraud_indices:

        fraud_type = random.choice(
            FRAUD_TYPES
        )

        if fraud_type == "ACCOUNT_TAKEOVER":

            inject_account_takeover(
                df,
                index
            )

        elif fraud_type == "HIGH_VALUE_FRAUD":

            inject_high_value_fraud(
                df,
                index
            )

        elif fraud_type == "GEOLOCATION_ANOMALY":

            inject_geolocation_anomaly(
                df,
                index
            )

        elif fraud_type == "VELOCITY_FRAUD":

            inject_velocity_fraud(
                df,
                index
            )

    return df


# ---------------------------------------------------------
# DATA QUALITY VALIDATION
# ---------------------------------------------------------

def run_quality_checks(df):
    """
    Run basic quality checks on the resulting dataset.
    """

    print()
    print("Data Quality Checks:")

    duplicate_ids = (
        df["transaction_id"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate transaction IDs:",
        duplicate_ids
    )

    missing_accounts = (
        df["account_id"]
        .isna()
        .sum()
    )

    print(
        "Missing account IDs:",
        missing_accounts
    )

    missing_customers = (
        df["customer_id"]
        .isna()
        .sum()
    )

    print(
        "Missing customer IDs:",
        missing_customers
    )

    invalid_fraud_labels = (
        ~df["is_fraud"].isin([0, 1])
    ).sum()

    print(
        "Invalid fraud labels:",
        invalid_fraud_labels
    )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":

    input_path = (
        "data/sample/transactions.csv"
    )

    output_path = (
        "data/sample/transactions_with_fraud.csv"
    )

    print(
        "Loading transaction dataset..."
    )

    transactions_df = pd.read_csv(
        input_path
    )

    print(
        f"Loaded {len(transactions_df)} transactions"
    )

    print()
    print(
        "Injecting fraud scenarios..."
    )

    fraud_df = inject_fraud(
        transactions_df
    )

    # Sort by timestamp
    fraud_df = fraud_df.sort_values(
        by="timestamp"
    )

    # Save final dataset
    fraud_df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    total_transactions = len(
        fraud_df
    )

    fraud_transactions = int(
        fraud_df["is_fraud"].sum()
    )

    normal_transactions = (
        total_transactions
        - fraud_transactions
    )

    fraud_percentage = (
        fraud_transactions
        / total_transactions
        * 100
    )

    print()
    print(
        f"Total transactions: {total_transactions}"
    )

    print(
        f"Normal transactions: {normal_transactions}"
    )

    print(
        f"Fraud transactions: {fraud_transactions}"
    )

    print(
        f"Fraud rate: {fraud_percentage:.2f}%"
    )

    print(
        f"Saved dataset to: {output_path}"
    )

    # -----------------------------------------------------
    # FRAUD DISTRIBUTION
    # -----------------------------------------------------

    print()
    print("Fraud Type Distribution:")

    fraud_distribution = (
        fraud_df[
            fraud_df["is_fraud"] == 1
        ]["fraud_type"]
        .value_counts()
    )

    print(
        fraud_distribution
    )

    # -----------------------------------------------------
    # RISK INDICATOR DISTRIBUTION
    # -----------------------------------------------------

    print()
    print("Risk Indicator Counts:")

    print(
        "New device:",
        fraud_df["new_device_flag"].sum()
    )

    print(
        "High amount:",
        fraud_df["high_amount_flag"].sum()
    )

    print(
        "Geolocation anomaly:",
        fraud_df["geo_anomaly_flag"].sum()
    )

    print(
        "Velocity:",
        fraud_df["velocity_flag"].sum()
    )

    # -----------------------------------------------------
    # QUALITY CHECKS
    # -----------------------------------------------------

    run_quality_checks(
        fraud_df
    )

    # -----------------------------------------------------
    # SAMPLE FRAUD RECORDS
    # -----------------------------------------------------

    print()
    print("Sample Fraud Transactions:")

    fraud_samples = fraud_df[
        fraud_df["is_fraud"] == 1
    ][
        [
            "transaction_id",
            "customer_id",
            "account_id",
            "amount",
            "channel",
            "country",
            "is_fraud",
            "fraud_type",
            "new_device_flag",
            "high_amount_flag",
            "geo_anomaly_flag",
            "velocity_flag",
        ]
    ].head(10)

    print(
        fraud_samples.to_string(
            index=False
        )
    )