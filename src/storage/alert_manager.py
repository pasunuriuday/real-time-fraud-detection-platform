"""
Fraud Alert Management
======================

Manages fraud investigation alerts stored in PostgreSQL.

Alert lifecycle:

NEW
    -> INVESTIGATING
        -> CONFIRMED_FRAUD
        -> FALSE_POSITIVE
        -> CLOSED

This module supports:

1. Creating fraud alerts
2. Preventing duplicate alerts
3. Retrieving alerts
4. Filtering alerts by status
5. Updating investigation status
6. Assigning analysts
7. Adding analyst notes
8. Tracking resolution timestamps
9. Alert statistics
"""

import os
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv(
    "FRAUD_DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "FRAUD_DB_PORT",
        "5432",
    )
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
# VALID ALERT STATUSES
# ============================================================

VALID_STATUSES = {
    "NEW",
    "INVESTIGATING",
    "CONFIRMED_FRAUD",
    "FALSE_POSITIVE",
    "CLOSED",
}


RESOLVED_STATUSES = {
    "CONFIRMED_FRAUD",
    "FALSE_POSITIVE",
    "CLOSED",
}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create a PostgreSQL connection.

    dict_row makes query results behave like dictionaries.
    """

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        row_factory=dict_row,
    )


# ============================================================
# CREATE FRAUD ALERT
# ============================================================

def create_fraud_alert(
    transaction,
    fraud_probability,
    risk_level,
):
    """
    Create a NEW fraud investigation alert.

    transaction_id is UNIQUE in fraud_alerts.

    Therefore:

    True  = new alert created
    False = alert already exists
    """

    transaction_id = transaction.get(
        "transaction_id"
    )

    if not transaction_id:

        raise ValueError(
            "transaction_id is required "
            "to create a fraud alert."
        )

    customer_id = transaction.get(
        "customer_id",
        "UNKNOWN",
    )

    account_id = transaction.get(
        "account_id",
        "UNKNOWN",
    )

    fraud_type = transaction.get(
        "fraud_type",
        "UNKNOWN",
    )

    query = """
        INSERT INTO fraud_alerts (
            transaction_id,
            customer_id,
            account_id,
            fraud_probability,
            risk_level,
            fraud_type,
            alert_status
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            'NEW'
        )

        ON CONFLICT (transaction_id)
        DO NOTHING

        RETURNING alert_id;
    """

    connection = None

    try:

        connection = get_connection()

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    transaction_id,
                    customer_id,
                    account_id,
                    float(fraud_probability),
                    risk_level,
                    fraud_type,
                ),
            )

            result = cursor.fetchone()

        connection.commit()

        return result is not None

    except Exception:

        if connection is not None:

            connection.rollback()

        raise

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# GET ONE ALERT
# ============================================================

def get_alert(
    transaction_id,
):
    """
    Retrieve one alert using transaction_id.

    Returns:
        dict -> alert found
        None -> alert not found
    """

    query = """
        SELECT
            alert_id,
            transaction_id,
            customer_id,
            account_id,
            fraud_probability,
            risk_level,
            fraud_type,
            alert_status,
            analyst_name,
            analyst_notes,
            created_at,
            updated_at,
            resolved_at

        FROM fraud_alerts

        WHERE transaction_id = %s;
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (transaction_id,),
            )

            result = cursor.fetchone()

            if result is None:

                return None

            return dict(result)

    finally:

        connection.close()


# ============================================================
# GET ALERT BY ID
# ============================================================

def get_alert_by_id(
    alert_id,
):
    """
    Retrieve one alert using alert_id.
    """

    query = """
        SELECT
            alert_id,
            transaction_id,
            customer_id,
            account_id,
            fraud_probability,
            risk_level,
            fraud_type,
            alert_status,
            analyst_name,
            analyst_notes,
            created_at,
            updated_at,
            resolved_at

        FROM fraud_alerts

        WHERE alert_id = %s;
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (alert_id,),
            )

            result = cursor.fetchone()

            if result is None:

                return None

            return dict(result)

    finally:

        connection.close()


# ============================================================
# GET ALERTS
# ============================================================

def get_alerts(
    status=None,
    limit=100,
):
    """
    Retrieve fraud alerts.

    Optional:
        status -> filter by investigation status
        limit  -> maximum number of rows
    """

    limit = int(limit)

    if limit <= 0:

        raise ValueError(
            "limit must be greater than zero."
        )

    if status is not None:

        status = status.upper()

        if status not in VALID_STATUSES:

            raise ValueError(
                f"Invalid alert status: {status}"
            )

        query = """
            SELECT
                alert_id,
                transaction_id,
                customer_id,
                account_id,
                fraud_probability,
                risk_level,
                fraud_type,
                alert_status,
                analyst_name,
                analyst_notes,
                created_at,
                updated_at,
                resolved_at

            FROM fraud_alerts

            WHERE alert_status = %s

            ORDER BY
                fraud_probability DESC,
                created_at DESC

            LIMIT %s;
        """

        parameters = (
            status,
            limit,
        )

    else:

        query = """
            SELECT
                alert_id,
                transaction_id,
                customer_id,
                account_id,
                fraud_probability,
                risk_level,
                fraud_type,
                alert_status,
                analyst_name,
                analyst_notes,
                created_at,
                updated_at,
                resolved_at

            FROM fraud_alerts

            ORDER BY
                fraud_probability DESC,
                created_at DESC

            LIMIT %s;
        """

        parameters = (
            limit,
        )

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                parameters,
            )

            rows = cursor.fetchall()

            return [
                dict(row)
                for row in rows
            ]

    finally:

        connection.close()


# ============================================================
# UPDATE ALERT
# ============================================================

def update_alert(
    transaction_id,
    status,
    analyst_name=None,
    analyst_notes=None,
):
    """
    Update alert investigation information.

    Returns:
        True  = alert updated
        False = alert not found
    """

    if not transaction_id:

        raise ValueError(
            "transaction_id is required."
        )

    if not status:

        raise ValueError(
            "status is required."
        )

    status = status.upper()

    if status not in VALID_STATUSES:

        raise ValueError(
            f"Invalid alert status: {status}"
        )

    if status in RESOLVED_STATUSES:

        resolved_at = datetime.now(
            timezone.utc
        )

    else:

        resolved_at = None

    query = """
        UPDATE fraud_alerts

        SET
            alert_status = %s,
            analyst_name = %s,
            analyst_notes = %s,
            updated_at = CURRENT_TIMESTAMP,
            resolved_at = %s

        WHERE transaction_id = %s

        RETURNING alert_id;
    """

    connection = None

    try:

        connection = get_connection()

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    status,
                    analyst_name,
                    analyst_notes,
                    resolved_at,
                    transaction_id,
                ),
            )

            result = cursor.fetchone()

        connection.commit()

        return result is not None

    except Exception:

        if connection is not None:

            connection.rollback()

        raise

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# ASSIGN ALERT TO ANALYST
# ============================================================

def assign_alert(
    transaction_id,
    analyst_name,
):
    """
    Assign an alert to an analyst.

    The alert automatically becomes INVESTIGATING.
    """

    if not analyst_name:

        raise ValueError(
            "analyst_name is required."
        )

    query = """
        UPDATE fraud_alerts

        SET
            alert_status = 'INVESTIGATING',
            analyst_name = %s,
            updated_at = CURRENT_TIMESTAMP,
            resolved_at = NULL

        WHERE transaction_id = %s

        RETURNING alert_id;
    """

    connection = None

    try:

        connection = get_connection()

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    analyst_name,
                    transaction_id,
                ),
            )

            result = cursor.fetchone()

        connection.commit()

        return result is not None

    except Exception:

        if connection is not None:

            connection.rollback()

        raise

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# ADD / UPDATE ANALYST NOTES
# ============================================================

def update_analyst_notes(
    transaction_id,
    analyst_notes,
):
    """
    Update analyst investigation notes without changing
    the current alert status.
    """

    query = """
        UPDATE fraud_alerts

        SET
            analyst_notes = %s,
            updated_at = CURRENT_TIMESTAMP

        WHERE transaction_id = %s

        RETURNING alert_id;
    """

    connection = None

    try:

        connection = get_connection()

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    analyst_notes,
                    transaction_id,
                ),
            )

            result = cursor.fetchone()

        connection.commit()

        return result is not None

    except Exception:

        if connection is not None:

            connection.rollback()

        raise

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# ALERT STATISTICS
# ============================================================

def get_alert_statistics():
    """
    Return alert queue statistics.
    """

    query = """
        SELECT

            COUNT(*) AS total_alerts,

            COUNT(*) FILTER (
                WHERE alert_status = 'NEW'
            ) AS new_alerts,

            COUNT(*) FILTER (
                WHERE alert_status = 'INVESTIGATING'
            ) AS investigating_alerts,

            COUNT(*) FILTER (
                WHERE alert_status = 'CONFIRMED_FRAUD'
            ) AS confirmed_fraud,

            COUNT(*) FILTER (
                WHERE alert_status = 'FALSE_POSITIVE'
            ) AS false_positives,

            COUNT(*) FILTER (
                WHERE alert_status = 'CLOSED'
            ) AS closed_alerts,

            COUNT(*) FILTER (
                WHERE risk_level = 'CRITICAL'
            ) AS critical_alerts,

            COUNT(*) FILTER (
                WHERE risk_level = 'HIGH'
            ) AS high_risk_alerts

        FROM fraud_alerts;
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(query)

            result = cursor.fetchone()

            return dict(result)

    finally:

        connection.close()


# ============================================================
# DELETE ALERT
# ============================================================

def delete_alert(
    transaction_id,
):
    """
    Delete an alert.

    Intended mainly for development/testing.
    """

    query = """
        DELETE FROM fraud_alerts

        WHERE transaction_id = %s

        RETURNING alert_id;
    """

    connection = None

    try:

        connection = get_connection()

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (transaction_id,),
            )

            result = cursor.fetchone()

        connection.commit()

        return result is not None

    except Exception:

        if connection is not None:

            connection.rollback()

        raise

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# TEST ALERT MANAGER
# ============================================================

def run_test():

    print("=" * 70)
    print("FRAUD ALERT MANAGER TEST")
    print("=" * 70)

    try:

        stats = get_alert_statistics()

        print()
        print(
            "PostgreSQL connection successful."
        )

        print()

        print(
            f"Total alerts       : "
            f"{stats['total_alerts']}"
        )

        print(
            f"New                : "
            f"{stats['new_alerts']}"
        )

        print(
            f"Investigating      : "
            f"{stats['investigating_alerts']}"
        )

        print(
            f"Confirmed fraud    : "
            f"{stats['confirmed_fraud']}"
        )

        print(
            f"False positives    : "
            f"{stats['false_positives']}"
        )

        print(
            f"Closed             : "
            f"{stats['closed_alerts']}"
        )

        print(
            f"Critical alerts    : "
            f"{stats['critical_alerts']}"
        )

        print(
            f"High-risk alerts   : "
            f"{stats['high_risk_alerts']}"
        )

        print()

        alerts = get_alerts(
            limit=10
        )

        print("=" * 70)
        print("CURRENT ALERTS")
        print("=" * 70)

        if not alerts:

            print()
            print(
                "No fraud alerts currently exist."
            )

        else:

            for alert in alerts:

                print()

                print(
                    f"Alert ID     : "
                    f"{alert['alert_id']}"
                )

                print(
                    f"Transaction  : "
                    f"{alert['transaction_id']}"
                )

                print(
                    f"Probability  : "
                    f"{float(alert['fraud_probability']):.2%}"
                )

                print(
                    f"Risk Level   : "
                    f"{alert['risk_level']}"
                )

                print(
                    f"Status       : "
                    f"{alert['alert_status']}"
                )

                print(
                    f"Fraud Type   : "
                    f"{alert['fraud_type']}"
                )

                print("-" * 70)

        print()
        print("=" * 70)
        print(
            "ALERT MANAGER TEST COMPLETED"
        )
        print("=" * 70)

    except Exception as error:

        print()
        print("=" * 70)
        print(
            "ALERT MANAGER TEST FAILED"
        )
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        raise


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_test()