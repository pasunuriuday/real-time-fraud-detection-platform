import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaException

from src.storage.prediction_writer import (
    initialize_prediction_storage,
    write_prediction,
)

from src.storage.postgres_writer import (
    test_connection,
    write_prediction_to_postgres,
)


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "banking-transactions"

# PostgreSQL-enabled inference consumer
CONSUMER_GROUP = "fraud-ml-postgres-v2"

MODEL_PATH = Path("models/fraud_random_forest.joblib")

# Optimized fraud classification threshold.
# Threshold analysis on the current labeled evaluation sample
# showed that 0.40 detected all four known fraud transactions.
FRAUD_THRESHOLD = 0.40


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("=" * 75)
    print("REAL-TIME FRAUD DETECTION - ML INFERENCE")
    print("=" * 75)

    print()
    print(f"Loading model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run src/ml/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    print("Fraud detection model loaded successfully.")

    return model


# ============================================================
# CREATE KAFKA CONSUMER
# ============================================================

def create_consumer():

    print()
    print("Connecting to Kafka...")

    config = {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }

    consumer = Consumer(config)

    consumer.subscribe([KAFKA_TOPIC])

    print(f"Kafka broker   : {KAFKA_BROKER}")
    print(f"Kafka topic    : {KAFKA_TOPIC}")
    print(f"Consumer group : {CONSUMER_GROUP}")

    return consumer


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_model_features(transaction):

    """
    Convert one Kafka transaction into the exact feature
    structure expected by the trained ML pipeline.
    """

    df = pd.DataFrame([transaction])

    required_columns = [
        "timestamp",
        "amount",
        "latitude",
        "longitude",
        "transaction_type",
        "channel",
        "merchant_category",
        "country",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Transaction is missing required fields: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # NUMERIC CLEANING
    # --------------------------------------------------------

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # TIMESTAMP PROCESSING
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    df["transaction_hour"] = (
        df["timestamp"].dt.hour
    )

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

    # --------------------------------------------------------
    # AMOUNT FEATURES
    # --------------------------------------------------------

    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    df["large_transaction"] = (
        df["amount"] >= 5000
    ).astype(int)

    df["very_large_transaction"] = (
        df["amount"] >= 10000
    ).astype(int)

    # --------------------------------------------------------
    # LOCATION FEATURE
    # --------------------------------------------------------

    df["international_transaction"] = (
        df["country"]
        .fillna("UNKNOWN")
        .ne("US")
        .astype(int)
    )

    # --------------------------------------------------------
    # EXACT MODEL FEATURES
    # --------------------------------------------------------

    feature_columns = [
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
        "transaction_type",
        "channel",
        "merchant_category",
        "country",
    ]

    return df[feature_columns]


# ============================================================
# SCORE TRANSACTION
# ============================================================

def score_transaction(
    model,
    transaction,
):

    features = create_model_features(
        transaction
    )

    probabilities = model.predict_proba(
        features
    )

    fraud_probability = float(
        probabilities[0][1]
    )

    predicted_fraud = int(
        fraud_probability >= FRAUD_THRESHOLD
    )

    return (
        fraud_probability,
        predicted_fraud,
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    # Risk severity is intentionally separate from the
    # binary fraud-classification threshold.

    if probability >= 0.80:
        return "CRITICAL"

    if probability >= 0.50:
        return "HIGH"

    if probability >= 0.25:
        return "MEDIUM"

    return "LOW"


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(
    transaction,
    probability,
    predicted_fraud,
    risk_level,
):

    transaction_id = transaction.get(
        "transaction_id",
        "UNKNOWN",
    )

    customer_id = transaction.get(
        "customer_id",
        "UNKNOWN",
    )

    account_id = transaction.get(
        "account_id",
        "UNKNOWN",
    )

    amount = transaction.get(
        "amount",
        0,
    )

    channel = transaction.get(
        "channel",
        "UNKNOWN",
    )

    country = transaction.get(
        "country",
        "UNKNOWN",
    )

    actual_fraud = transaction.get(
        "is_fraud",
        "UNKNOWN",
    )

    fraud_type = transaction.get(
        "fraud_type",
        "UNKNOWN",
    )

    print()
    print("-" * 75)

    print(
        f"Transaction : {transaction_id}"
    )

    print(
        f"Customer    : {customer_id}"
    )

    print(
        f"Account     : {account_id}"
    )

    print(
        f"Amount      : ${float(amount):,.2f}"
    )

    print(
        f"Channel     : {channel}"
    )

    print(
        f"Country     : {country}"
    )

    print(
        f"Probability : {probability:.2%}"
    )

    print(
        f"Risk Level  : {risk_level}"
    )

    print(
        "Prediction  : "
        + (
            "FRAUD"
            if predicted_fraud
            else "NORMAL"
        )
    )

    print(
        f"Actual Label: {actual_fraud}"
    )

    print(
        f"Fraud Type  : {fraud_type}"
    )

    if predicted_fraud:

        print()
        print("!!! FRAUD ALERT !!!")

        print(
            f"Transaction {transaction_id} "
            f"exceeded the "
            f"{FRAUD_THRESHOLD:.0%} "
            "ML fraud threshold."
        )

    print("-" * 75)


# ============================================================
# PRINT PROCESSING SUMMARY
# ============================================================

def print_processing_summary(
    processed_count,
    fraud_alert_count,
    csv_count,
    postgres_count,
    duplicate_count,
    failed_count,
):

    print()
    print("=" * 55)

    print(
        f"Processed transactions : "
        f"{processed_count}"
    )

    print(
        f"Fraud alerts           : "
        f"{fraud_alert_count}"
    )

    print(
        f"CSV writes             : "
        f"{csv_count}"
    )

    print(
        f"PostgreSQL inserts     : "
        f"{postgres_count}"
    )

    print(
        f"Duplicate DB skips     : "
        f"{duplicate_count}"
    )

    print(
        f"Failed transactions    : "
        f"{failed_count}"
    )

    print("=" * 55)


# ============================================================
# REAL-TIME DETECTOR
# ============================================================

def run_detector():

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # INITIALIZE CSV STORAGE
    # --------------------------------------------------------

    initialize_prediction_storage()

    # --------------------------------------------------------
    # VERIFY POSTGRESQL
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CHECKING POSTGRESQL")
    print("=" * 75)

    test_connection()

    # --------------------------------------------------------
    # CREATE KAFKA CONSUMER
    # --------------------------------------------------------

    consumer = create_consumer()

    # --------------------------------------------------------
    # PROCESSING COUNTERS
    # --------------------------------------------------------

    processed_count = 0
    fraud_alert_count = 0
    postgres_count = 0
    duplicate_count = 0
    csv_count = 0
    failed_count = 0

    print()
    print("=" * 75)

    print(
        "LISTENING FOR REAL-TIME BANKING TRANSACTIONS"
    )

    print("=" * 75)

    print()
    print(
        "Press Control + C to stop the detector."
    )

    try:

        while True:

            # ------------------------------------------------
            # READ KAFKA MESSAGE
            # ------------------------------------------------

            message = consumer.poll(
                timeout=1.0
            )

            if message is None:
                continue

            if message.error():

                raise KafkaException(
                    message.error()
                )

            try:

                # --------------------------------------------
                # DECODE KAFKA MESSAGE
                # --------------------------------------------

                raw_value = (
                    message
                    .value()
                    .decode("utf-8")
                )

                transaction = json.loads(
                    raw_value
                )

                # --------------------------------------------
                # ML INFERENCE
                # --------------------------------------------

                (
                    fraud_probability,
                    predicted_fraud,
                ) = score_transaction(
                    model,
                    transaction,
                )

                # --------------------------------------------
                # RISK CLASSIFICATION
                # --------------------------------------------

                risk_level = get_risk_level(
                    fraud_probability
                )

                # --------------------------------------------
                # DISPLAY RESULT
                # --------------------------------------------

                display_prediction(
                    transaction=transaction,
                    probability=fraud_probability,
                    predicted_fraud=predicted_fraud,
                    risk_level=risk_level,
                )

                # --------------------------------------------
                # CSV STORAGE
                # --------------------------------------------

                write_prediction(
                    transaction=transaction,
                    fraud_probability=fraud_probability,
                    predicted_fraud=predicted_fraud,
                    risk_level=risk_level,
                )

                csv_count += 1

                # --------------------------------------------
                # POSTGRESQL STORAGE
                # --------------------------------------------

                inserted = write_prediction_to_postgres(
                    transaction=transaction,
                    fraud_probability=fraud_probability,
                    predicted_fraud=predicted_fraud,
                    risk_level=risk_level,
                )

                # Count actual inserts separately from
                # duplicate transactions skipped by PostgreSQL.
                if inserted:
                    postgres_count += 1
                else:
                    duplicate_count += 1

                # --------------------------------------------
                # UPDATE COUNTERS
                # --------------------------------------------

                processed_count += 1

                if predicted_fraud:
                    fraud_alert_count += 1

                # --------------------------------------------
                # COMMIT KAFKA OFFSET
                # --------------------------------------------

                # Offset is committed only after:
                #
                # 1. ML prediction succeeds
                # 2. CSV write succeeds
                # 3. PostgreSQL operation succeeds
                #
                # A duplicate PostgreSQL transaction is not
                # considered a processing failure.

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                # --------------------------------------------
                # PROGRESS REPORT
                # --------------------------------------------

                if processed_count % 10 == 0:

                    print_processing_summary(
                        processed_count=processed_count,
                        fraud_alert_count=fraud_alert_count,
                        csv_count=csv_count,
                        postgres_count=postgres_count,
                        duplicate_count=duplicate_count,
                        failed_count=failed_count,
                    )

            except Exception as error:

                failed_count += 1

                print()
                print("=" * 75)

                print(
                    "FAILED TO PROCESS TRANSACTION"
                )

                print("=" * 75)

                print(
                    f"Kafka partition: "
                    f"{message.partition()}"
                )

                print(
                    f"Kafka offset   : "
                    f"{message.offset()}"
                )

                print(
                    f"Error          : "
                    f"{error}"
                )

                print()

                print(
                    "Kafka offset NOT committed."
                )

                print("=" * 75)

    except KeyboardInterrupt:

        print()
        print()

        print("=" * 75)

        print(
            "STOPPING REAL-TIME FRAUD DETECTOR"
        )

        print("=" * 75)

        print(
            f"Transactions processed : "
            f"{processed_count}"
        )

        print(
            f"Fraud alerts generated : "
            f"{fraud_alert_count}"
        )

        print(
            f"CSV writes             : "
            f"{csv_count}"
        )

        print(
            f"PostgreSQL inserts     : "
            f"{postgres_count}"
        )

        print(
            f"Duplicate DB skips     : "
            f"{duplicate_count}"
        )

        print(
            f"Failed transactions    : "
            f"{failed_count}"
        )

        print("=" * 75)

    finally:

        consumer.close()

        print()
        print(
            "Kafka consumer closed."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_detector()