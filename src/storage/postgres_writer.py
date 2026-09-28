import os
from datetime import datetime, timezone

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
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a PostgreSQL connection.
    """

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# TEST DATABASE CONNECTION
# ============================================================

def test_connection():
    """
    Verify that PostgreSQL is reachable and that the
    fraud_predictions table exists.
    """

    print("=" * 70)
    print("POSTGRESQL CONNECTION TEST")
    print("=" * 70)

    try:

        with get_connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        current_database(),
                        current_user;
                    """
                )

                result = cursor.fetchone()

                print()
                print(f"Database : {result[0]}")
                print(f"User     : {result[1]}")

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM fraud_predictions;
                    """
                )

                count = cursor.fetchone()[0]

                print(
                    f"Existing predictions: {count}"
                )

        print()
        print(
            "PostgreSQL connection successful."
        )

        return True

    except Exception as error:

        print()
        print(
            "PostgreSQL connection failed."
        )

        print(
            f"Error: {error}"
        )

        raise


# ============================================================
# WRITE PREDICTION TO POSTGRESQL
# ============================================================

def write_prediction_to_postgres(
    transaction,
    fraud_probability,
    predicted_fraud,
    risk_level,
):
    """
    Insert one real-time fraud prediction
    into PostgreSQL.
    """

    prediction_timestamp = datetime.now(
        timezone.utc
    )

    query = """
        INSERT INTO fraud_predictions (
            transaction_id,
            customer_id,
            account_id,
            amount,
            channel,
            country,
            fraud_probability,
            predicted_fraud,
            risk_level,
            actual_fraud,
            fraud_type,
            prediction_timestamp
        )
        VALUES (
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s
)
ON CONFLICT (transaction_id)
DO NOTHING;
    """

    values = (
        transaction.get("transaction_id"),
        transaction.get("customer_id"),
        transaction.get("account_id"),
        transaction.get("amount"),
        transaction.get("channel"),
        transaction.get("country"),
        float(fraud_probability),
        int(predicted_fraud),
        risk_level,
        transaction.get("is_fraud"),
        transaction.get("fraud_type"),
        prediction_timestamp,
    )

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                values,
            )

        connection.commit()

    return prediction_timestamp


# ============================================================
# DIRECT CONNECTION TEST
# ============================================================

if __name__ == "__main__":

    test_connection()