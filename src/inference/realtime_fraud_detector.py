import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaException


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "banking-transactions"

CONSUMER_GROUP = "fraud-ml-inference-v1"

MODEL_PATH = Path(
    "models/fraud_random_forest.joblib"
)

# Start with the standard classifier threshold.
# Later we will tune this using precision/recall.
FRAUD_THRESHOLD = 0.50


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

        # For a brand-new consumer group, begin with the
        # earliest available messages.
        "auto.offset.reset": "earliest",

        # We commit only after successfully processing
        # a transaction.
        "enable.auto.commit": False,
    }

    consumer = Consumer(config)

    consumer.subscribe(
        [KAFKA_TOPIC]
    )

    print(f"Kafka broker : {KAFKA_BROKER}")
    print(f"Kafka topic  : {KAFKA_TOPIC}")
    print(f"Consumer group: {CONSUMER_GROUP}")

    return consumer


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_model_features(transaction):

    """
    Convert one Kafka transaction into the exact feature
    structure expected by the trained sklearn pipeline.
    """

    df = pd.DataFrame(
        [transaction]
    )

    # --------------------------------------------------------
    # REQUIRED RAW COLUMNS
    # --------------------------------------------------------

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
    # CLEAN NUMERIC VALUES
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
    # TIMESTAMP FEATURES
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

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
    # EXACT MODEL FEATURE SET
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

    fraud_probability = float(
        model.predict_proba(
            features
        )[0][1]
    )

    predicted_fraud = int(
        fraud_probability >= FRAUD_THRESHOLD
    )

    return (
        fraud_probability,
        predicted_fraud,
    )


# ============================================================
# DETERMINE DISPLAY RISK LEVEL
# ============================================================

def get_risk_level(probability):

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
):

    transaction_id = transaction.get(
        "transaction_id",
        "UNKNOWN"
    )

    customer_id = transaction.get(
        "customer_id",
        "UNKNOWN"
    )

    account_id = transaction.get(
        "account_id",
        "UNKNOWN"
    )

    amount = transaction.get(
        "amount",
        0
    )

    channel = transaction.get(
        "channel",
        "UNKNOWN"
    )

    country = transaction.get(
        "country",
        "UNKNOWN"
    )

    actual_fraud = transaction.get(
        "is_fraud",
        "UNKNOWN"
    )

    fraud_type = transaction.get(
        "fraud_type",
        "UNKNOWN"
    )

    risk_level = get_risk_level(
        probability
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
        f"Prediction  : "
        f"{'FRAUD' if predicted_fraud else 'NORMAL'}"
    )

    # These are shown only for evaluation because our
    # synthetic Kafka data contains ground-truth labels.
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
            f"{FRAUD_THRESHOLD:.0%} ML threshold."
        )

    print("-" * 75)


# ============================================================
# REAL-TIME CONSUMER LOOP
# ============================================================

def run_detector():

    model = load_model()

    consumer = create_consumer()

    processed_count = 0
    fraud_alert_count = 0

    print()
    print("=" * 75)
    print("LISTENING FOR REAL-TIME BANKING TRANSACTIONS")
    print("=" * 75)

    print()
    print(
        "Press Control + C to stop the detector."
    )

    try:

        while True:

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

                raw_value = (
                    message.value()
                    .decode("utf-8")
                )

                transaction = json.loads(
                    raw_value
                )

                (
                    fraud_probability,
                    predicted_fraud,
                ) = score_transaction(
                    model,
                    transaction,
                )

                display_prediction(
                    transaction,
                    fraud_probability,
                    predicted_fraud,
                )

                processed_count += 1

                if predicted_fraud:
                    fraud_alert_count += 1

                # Commit after successful processing.
                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                if processed_count % 10 == 0:

                    print()
                    print(
                        f"Processed transactions : "
                        f"{processed_count}"
                    )

                    print(
                        f"Fraud alerts           : "
                        f"{fraud_alert_count}"
                    )

            except Exception as error:

                print()
                print(
                    "Failed to process transaction:"
                )

                print(error)

                # Do not commit this message because
                # processing failed.

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 75)
        print("STOPPING REAL-TIME FRAUD DETECTOR")
        print("=" * 75)

        print(
            f"Transactions processed: "
            f"{processed_count}"
        )

        print(
            f"Fraud alerts generated: "
            f"{fraud_alert_count}"
        )

    finally:

        consumer.close()

        print()
        print("Kafka consumer closed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_detector()