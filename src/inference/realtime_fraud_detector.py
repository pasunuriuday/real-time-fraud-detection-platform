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

from src.storage.alert_manager import (
    create_fraud_alert,
)


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "banking-transactions"

CONSUMER_GROUP = "fraud-ml-alerts-v2"

MODEL_PATH = Path(
    "models/fraud_random_forest.joblib"
)

# Optimized fraud threshold
FRAUD_THRESHOLD = 0.40


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 75)
    print(
        "REAL-TIME FRAUD DETECTION - ML INFERENCE"
    )
    print("=" * 75)

    print()
    print(
        f"Loading model: {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Fraud detection model loaded successfully."
    )

    return model


# ============================================================
# CREATE KAFKA CONSUMER
# ============================================================

def create_consumer():

    print()
    print(
        "Connecting to Kafka..."
    )

    config = {

        "bootstrap.servers":
            KAFKA_BROKER,

        "group.id":
            CONSUMER_GROUP,

        "auto.offset.reset":
            "earliest",

        "enable.auto.commit":
            False,
    }

    consumer = Consumer(
        config
    )

    consumer.subscribe(
        [KAFKA_TOPIC]
    )

    print(
        f"Kafka broker   : "
        f"{KAFKA_BROKER}"
    )

    print(
        f"Kafka topic    : "
        f"{KAFKA_TOPIC}"
    )

    print(
        f"Consumer group : "
        f"{CONSUMER_GROUP}"
    )

    return consumer


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_model_features(
    transaction,
):

    df = pd.DataFrame(
        [transaction]
    )

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

        for column
        in required_columns

        if column
        not in df.columns
    ]

    if missing_columns:

        raise ValueError(

            "Transaction is missing "
            "required fields: "

            + ", ".join(
                missing_columns
            )
        )

    # ========================================================
    # NUMERIC CLEANING
    # ========================================================

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

    # ========================================================
    # TIMESTAMP
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid transaction timestamp."
        )

    # ========================================================
    # TIME FEATURES
    # ========================================================

    df["transaction_hour"] = (
        df["timestamp"]
        .dt
        .hour
    )

    df[
        "transaction_day_of_week"
    ] = (
        df["timestamp"]
        .dt
        .dayofweek
    )

    df["is_weekend"] = (

        df[
            "transaction_day_of_week"
        ]

        .isin(
            [5, 6]
        )

        .astype(int)
    )

    df["night_transaction"] = (

        df["transaction_hour"]

        .between(
            0,
            4,
        )

        .astype(int)
    )

    # ========================================================
    # AMOUNT FEATURES
    # ========================================================

    df["log_amount"] = np.log1p(

        df["amount"]
        .clip(
            lower=0
        )
    )

    df["large_transaction"] = (

        df["amount"]
        >= 5000

    ).astype(int)

    df[
        "very_large_transaction"
    ] = (

        df["amount"]
        >= 10000

    ).astype(int)

    # ========================================================
    # LOCATION FEATURE
    # ========================================================

    df[
        "international_transaction"
    ] = (

        df["country"]

        .fillna(
            "UNKNOWN"
        )

        .ne(
            "US"
        )

        .astype(int)
    )

    # ========================================================
    # EXACT MODEL FEATURES
    # ========================================================

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

    return df[
        feature_columns
    ]


# ============================================================
# SCORE TRANSACTION
# ============================================================

def score_transaction(
    model,
    transaction,
):

    features = (
        create_model_features(
            transaction
        )
    )

    probabilities = (
        model.predict_proba(
            features
        )
    )

    fraud_probability = float(
        probabilities[0][1]
    )

    predicted_fraud = int(

        fraud_probability
        >= FRAUD_THRESHOLD
    )

    return (
        fraud_probability,
        predicted_fraud,
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk_level(
    probability,
):

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

    transaction_id = (
        transaction.get(
            "transaction_id",
            "UNKNOWN",
        )
    )

    customer_id = (
        transaction.get(
            "customer_id",
            "UNKNOWN",
        )
    )

    account_id = (
        transaction.get(
            "account_id",
            "UNKNOWN",
        )
    )

    amount = (
        transaction.get(
            "amount",
            0,
        )
    )

    channel = (
        transaction.get(
            "channel",
            "UNKNOWN",
        )
    )

    country = (
        transaction.get(
            "country",
            "UNKNOWN",
        )
    )

    actual_fraud = (
        transaction.get(
            "is_fraud",
            "UNKNOWN",
        )
    )

    fraud_type = (
        transaction.get(
            "fraud_type",
            "UNKNOWN",
        )
    )

    print()
    print("-" * 75)

    print(
        f"Transaction : "
        f"{transaction_id}"
    )

    print(
        f"Customer    : "
        f"{customer_id}"
    )

    print(
        f"Account     : "
        f"{account_id}"
    )

    print(
        f"Amount      : "
        f"${float(amount):,.2f}"
    )

    print(
        f"Channel     : "
        f"{channel}"
    )

    print(
        f"Country     : "
        f"{country}"
    )

    print(
        f"Probability : "
        f"{probability:.2%}"
    )

    print(
        f"Risk Level  : "
        f"{risk_level}"
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
        f"Actual Label: "
        f"{actual_fraud}"
    )

    print(
        f"Fraud Type  : "
        f"{fraud_type}"
    )

    if predicted_fraud:

        print()

        print(
            "!!! FRAUD ALERT !!!"
        )

        print(

            f"Transaction "
            f"{transaction_id} "

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
    fraud_prediction_count,
    csv_insert_count,
    csv_duplicate_count,
    postgres_insert_count,
    postgres_duplicate_count,
    alert_created_count,
    alert_duplicate_count,
    failed_count,
):

    print()
    print("=" * 65)

    print(
        f"Processed transactions : "
        f"{processed_count}"
    )

    print(
        f"Fraud predictions      : "
        f"{fraud_prediction_count}"
    )

    print(
        f"CSV inserts            : "
        f"{csv_insert_count}"
    )

    print(
        f"Duplicate CSV skips    : "
        f"{csv_duplicate_count}"
    )

    print(
        f"PostgreSQL inserts     : "
        f"{postgres_insert_count}"
    )

    print(
        f"Duplicate DB skips     : "
        f"{postgres_duplicate_count}"
    )

    print(
        f"Alerts created         : "
        f"{alert_created_count}"
    )

    print(
        f"Duplicate alert skips  : "
        f"{alert_duplicate_count}"
    )

    print(
        f"Failed transactions    : "
        f"{failed_count}"
    )

    print("=" * 65)


# ============================================================
# REAL-TIME DETECTOR
# ============================================================

def run_detector():

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # CSV STORAGE
    # ========================================================

    initialize_prediction_storage()

    # ========================================================
    # POSTGRESQL
    # ========================================================

    print()
    print("=" * 75)

    print(
        "CHECKING POSTGRESQL"
    )

    print("=" * 75)

    test_connection()

    # ========================================================
    # KAFKA
    # ========================================================

    consumer = (
        create_consumer()
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    processed_count = 0

    fraud_prediction_count = 0

    csv_insert_count = 0

    csv_duplicate_count = 0

    postgres_insert_count = 0

    postgres_duplicate_count = 0

    alert_created_count = 0

    alert_duplicate_count = 0

    failed_count = 0

    # ========================================================
    # START LISTENER
    # ========================================================

    print()
    print("=" * 75)

    print(
        "LISTENING FOR REAL-TIME "
        "BANKING TRANSACTIONS"
    )

    print("=" * 75)

    print()

    print(
        "Press Control + C "
        "to stop the detector."
    )

    try:

        while True:

            # =================================================
            # READ KAFKA MESSAGE
            # =================================================

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

                # =============================================
                # DECODE MESSAGE
                # =============================================

                raw_value = (

                    message
                    .value()
                    .decode(
                        "utf-8"
                    )
                )

                transaction = (
                    json.loads(
                        raw_value
                    )
                )

                transaction_id = (
                    transaction.get(
                        "transaction_id"
                    )
                )

                if not transaction_id:

                    raise ValueError(
                        "transaction_id "
                        "is required."
                    )

                # =============================================
                # ML INFERENCE
                # =============================================

                (
                    fraud_probability,
                    predicted_fraud,

                ) = score_transaction(

                    model,
                    transaction,
                )

                # =============================================
                # RISK CLASSIFICATION
                # =============================================

                risk_level = (
                    get_risk_level(
                        fraud_probability
                    )
                )

                # =============================================
                # DISPLAY RESULT
                # =============================================

                display_prediction(

                    transaction=transaction,

                    probability=(
                        fraud_probability
                    ),

                    predicted_fraud=(
                        predicted_fraud
                    ),

                    risk_level=(
                        risk_level
                    ),
                )

                # =============================================
                # CSV STORAGE
                # =============================================

                csv_inserted = (
                    write_prediction(

                        transaction=transaction,

                        fraud_probability=(
                            fraud_probability
                        ),

                        predicted_fraud=(
                            predicted_fraud
                        ),

                        risk_level=(
                            risk_level
                        ),
                    )
                )

                if csv_inserted:

                    csv_insert_count += 1

                else:

                    csv_duplicate_count += 1

                # =============================================
                # POSTGRESQL STORAGE
                # =============================================

                postgres_inserted = (
                    write_prediction_to_postgres(

                        transaction=transaction,

                        fraud_probability=(
                            fraud_probability
                        ),

                        predicted_fraud=(
                            predicted_fraud
                        ),

                        risk_level=(
                            risk_level
                        ),
                    )
                )

                if postgres_inserted:

                    postgres_insert_count += 1

                else:

                    postgres_duplicate_count += 1

                # =============================================
                # FRAUD ALERT MANAGEMENT
                # =============================================

                if predicted_fraud:

                    fraud_prediction_count += 1

                    alert_created = (
                        create_fraud_alert(

                            transaction=transaction,

                            fraud_probability=(
                                fraud_probability
                            ),

                            risk_level=(
                                risk_level
                            ),
                        )
                    )

                    if alert_created:

                        alert_created_count += 1

                        print()

                        print(
                            "INVESTIGATION ALERT "
                            "CREATED"
                        )

                        print(
                            f"Transaction : "
                            f"{transaction_id}"
                        )

                        print(
                            f"Risk        : "
                            f"{risk_level}"
                        )

                        print(
                            f"Probability : "
                            f"{fraud_probability:.2%}"
                        )

                    else:

                        alert_duplicate_count += 1

                        print()

                        print(
                            "Investigation alert "
                            "already exists - skipped."
                        )

                # =============================================
                # MARK SUCCESSFUL PROCESSING
                # =============================================

                processed_count += 1

                # =============================================
                # COMMIT KAFKA OFFSET
                # =============================================

                # Commit happens only after:
                #
                # 1. ML inference succeeds
                # 2. CSV operation succeeds
                # 3. PostgreSQL operation succeeds
                # 4. Fraud alert operation succeeds
                #    when the prediction is fraud

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                # =============================================
                # PROGRESS REPORT
                # =============================================

                if (
                    processed_count
                    % 10
                    == 0
                ):

                    print_processing_summary(

                        processed_count=(
                            processed_count
                        ),

                        fraud_prediction_count=(
                            fraud_prediction_count
                        ),

                        csv_insert_count=(
                            csv_insert_count
                        ),

                        csv_duplicate_count=(
                            csv_duplicate_count
                        ),

                        postgres_insert_count=(
                            postgres_insert_count
                        ),

                        postgres_duplicate_count=(
                            postgres_duplicate_count
                        ),

                        alert_created_count=(
                            alert_created_count
                        ),

                        alert_duplicate_count=(
                            alert_duplicate_count
                        ),

                        failed_count=(
                            failed_count
                        ),
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
            "STOPPING REAL-TIME "
            "FRAUD DETECTOR"
        )

        print("=" * 75)

        print_processing_summary(

            processed_count=(
                processed_count
            ),

            fraud_prediction_count=(
                fraud_prediction_count
            ),

            csv_insert_count=(
                csv_insert_count
            ),

            csv_duplicate_count=(
                csv_duplicate_count
            ),

            postgres_insert_count=(
                postgres_insert_count
            ),

            postgres_duplicate_count=(
                postgres_duplicate_count
            ),

            alert_created_count=(
                alert_created_count
            ),

            alert_duplicate_count=(
                alert_duplicate_count
            ),

            failed_count=(
                failed_count
            ),
        )

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