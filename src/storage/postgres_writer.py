import os
from datetime import datetime, timezone

import psycopg


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv(
    "FRAUD_DB_HOST",
    "localhost",
)

DB_PORT = os.getenv(
    "FRAUD_DB_PORT",
    "5432",
)

DB_NAME = os.getenv(
    "FRAUD_DB_NAME",
    "fraud_detection",
)

DB_USER = os.getenv(
    "FRAUD_DB_USER",
    "saleor",
)

DB_PASSWORD = os.getenv(
    "FRAUD_DB_PASSWORD",
    "saleor",
)


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
    Verify PostgreSQL connectivity and confirm that
    the fraud_predictions table is available.
    """

    print("=" * 70)
    print("POSTGRESQL CONNECTION TEST")
    print("=" * 70)

    try:

        with get_connection() as connection:

            with connection.cursor() as cursor:

                # --------------------------------------------
                # DATABASE INFORMATION
                # --------------------------------------------

                cursor.execute(
                    """
                    SELECT
                        current_database(),
                        current_user;
                    """
                )

                result = cursor.fetchone()

                print()
                print(
                    f"Database : {result[0]}"
                )

                print(
                    f"User     : {result[1]}"
                )

                # --------------------------------------------
                # PREDICTION COUNT
                # --------------------------------------------

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

                # --------------------------------------------
                # UNIQUE TRANSACTION COUNT
                # --------------------------------------------

                cursor.execute(
                    """
                    SELECT
                        COUNT(
                            DISTINCT transaction_id
                        )
                    FROM fraud_predictions;
                    """
                )

                unique_count = (
                    cursor.fetchone()[0]
                )

                print(
                    f"Unique transactions : "
                    f"{unique_count}"
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
    Insert one real-time fraud prediction into PostgreSQL.

    Returns
    -------
    True
        A new prediction was inserted.

    False
        transaction_id already existed and PostgreSQL
        skipped the duplicate.
    """

    # --------------------------------------------------------
    # VALIDATE TRANSACTION ID
    # --------------------------------------------------------

    transaction_id = transaction.get(
        "transaction_id"
    )

    if not transaction_id:

        raise ValueError(
            "transaction_id is required "
            "for PostgreSQL storage."
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    prediction_timestamp = datetime.now(
        timezone.utc
    )

    # --------------------------------------------------------
    # IDEMPOTENT INSERT
    # --------------------------------------------------------

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
        DO NOTHING

        RETURNING transaction_id;
    """

    values = (

        transaction_id,

        transaction.get(
            "customer_id"
        ),

        transaction.get(
            "account_id"
        ),

        transaction.get(
            "amount"
        ),

        transaction.get(
            "channel"
        ),

        transaction.get(
            "country"
        ),

        float(
            fraud_probability
        ),

        int(
            predicted_fraud
        ),

        risk_level,

        transaction.get(
            "is_fraud"
        ),

        transaction.get(
            "fraud_type"
        ),

        prediction_timestamp,
    )

    # --------------------------------------------------------
    # EXECUTE INSERT
    # --------------------------------------------------------

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                values,
            )

            result = cursor.fetchone()

        connection.commit()

    # --------------------------------------------------------
    # RETURN INSERT STATUS
    # --------------------------------------------------------

    # INSERT happened:
    #
    # RETURNING transaction_id
    # -> ('TXN-...',)
    #
    # Duplicate:
    #
    # ON CONFLICT DO NOTHING
    # -> None

    return result is not None


# ============================================================
# GET DATABASE STATISTICS
# ============================================================

def get_prediction_statistics():
    """
    Return basic prediction-storage statistics.
    """

    query = """
        SELECT

            COUNT(*) AS total_rows,

            COUNT(
                DISTINCT transaction_id
            ) AS unique_transactions,

            COUNT(*) FILTER (
                WHERE predicted_fraud = 1
            ) AS fraud_predictions,

            COUNT(*) FILTER (
                WHERE risk_level = 'CRITICAL'
            ) AS critical_risk,

            COUNT(*) FILTER (
                WHERE risk_level = 'HIGH'
            ) AS high_risk,

            COUNT(*) FILTER (
                WHERE risk_level = 'MEDIUM'
            ) AS medium_risk,

            COUNT(*) FILTER (
                WHERE risk_level = 'LOW'
            ) AS low_risk

        FROM fraud_predictions;
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(query)

            result = cursor.fetchone()

    return {
        "total_rows": result[0],
        "unique_transactions": result[1],
        "fraud_predictions": result[2],
        "critical_risk": result[3],
        "high_risk": result[4],
        "medium_risk": result[5],
        "low_risk": result[6],
    }


# ============================================================
# DIRECT CONNECTION TEST
# ============================================================

if __name__ == "__main__":

    test_connection()

    print()
    print("=" * 70)
    print("PREDICTION STORAGE STATISTICS")
    print("=" * 70)

    statistics = (
        get_prediction_statistics()
    )

    print()

    print(
        f"Total rows          : "
        f"{statistics['total_rows']}"
    )

    print(
        f"Unique transactions : "
        f"{statistics['unique_transactions']}"
    )

    print(
        f"Fraud predictions   : "
        f"{statistics['fraud_predictions']}"
    )

    print(
        f"Critical risk       : "
        f"{statistics['critical_risk']}"
    )

    print(
        f"High risk           : "
        f"{statistics['high_risk']}"
    )

    print(
        f"Medium risk         : "
        f"{statistics['medium_risk']}"
    )

    print(
        f"Low risk            : "
        f"{statistics['low_risk']}"
    )

    print()
    print("=" * 70)