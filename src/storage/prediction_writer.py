from pathlib import Path
from datetime import datetime, timezone
import csv


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("data/predictions")
PREDICTION_FILE = OUTPUT_DIR / "fraud_predictions.csv"


# ============================================================
# CSV COLUMNS
# ============================================================

FIELDNAMES = [
    "transaction_id",
    "customer_id",
    "account_id",
    "amount",
    "channel",
    "country",
    "fraud_probability",
    "predicted_fraud",
    "risk_level",
    "actual_fraud",
    "fraud_type",
    "prediction_timestamp",
]


# ============================================================
# DUPLICATE CACHE
# ============================================================

_existing_transaction_ids = set()


# ============================================================
# INITIALIZE STORAGE
# ============================================================

def initialize_prediction_storage():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create file if it does not exist
    if not PREDICTION_FILE.exists():

        with PREDICTION_FILE.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
            )

            writer.writeheader()

        print(
            f"Prediction storage initialized: "
            f"{PREDICTION_FILE}"
        )

    else:

        print(
            f"Prediction storage found: "
            f"{PREDICTION_FILE}"
        )

    # Reload existing IDs
    _existing_transaction_ids.clear()

    with PREDICTION_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            transaction_id = row.get(
                "transaction_id"
            )

            if transaction_id:

                _existing_transaction_ids.add(
                    transaction_id
                )

    print(
        f"Existing unique CSV transactions: "
        f"{len(_existing_transaction_ids)}"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def transaction_exists(transaction_id):

    return (
        transaction_id
        in _existing_transaction_ids
    )


# ============================================================
# CREATE RECORD
# ============================================================

def create_prediction_record(
    transaction,
    fraud_probability,
    predicted_fraud,
    risk_level,
):

    return {

        "transaction_id": transaction.get(
            "transaction_id",
            "UNKNOWN",
        ),

        "customer_id": transaction.get(
            "customer_id",
            "UNKNOWN",
        ),

        "account_id": transaction.get(
            "account_id",
            "UNKNOWN",
        ),

        "amount": transaction.get(
            "amount",
            0,
        ),

        "channel": transaction.get(
            "channel",
            "UNKNOWN",
        ),

        "country": transaction.get(
            "country",
            "UNKNOWN",
        ),

        "fraud_probability": round(
            float(fraud_probability),
            6,
        ),

        "predicted_fraud": int(
            predicted_fraud
        ),

        "risk_level": risk_level,

        "actual_fraud": transaction.get(
            "is_fraud",
            None,
        ),

        "fraud_type": transaction.get(
            "fraud_type",
            "UNKNOWN",
        ),

        "prediction_timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# ============================================================
# WRITE PREDICTION
# ============================================================

def write_prediction(
    transaction,
    fraud_probability,
    predicted_fraud,
    risk_level,
):

    transaction_id = transaction.get(
        "transaction_id"
    )

    if not transaction_id:

        raise ValueError(
            "transaction_id is required."
        )

    # --------------------------------------------------------
    # DUPLICATE PROTECTION
    # --------------------------------------------------------

    if transaction_exists(
        transaction_id
    ):

        return False

    # --------------------------------------------------------
    # BUILD RECORD
    # --------------------------------------------------------

    record = create_prediction_record(
        transaction=transaction,
        fraud_probability=fraud_probability,
        predicted_fraud=predicted_fraud,
        risk_level=risk_level,
    )

    # --------------------------------------------------------
    # WRITE RECORD
    # --------------------------------------------------------

    with PREDICTION_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writerow(record)

    # Only add to cache after successful write
    _existing_transaction_ids.add(
        transaction_id
    )

    return True


# ============================================================
# CSV STATISTICS
# ============================================================

def get_csv_transaction_count():

    return len(
        _existing_transaction_ids
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FRAUD PREDICTION STORAGE TEST")
    print("=" * 70)

    initialize_prediction_storage()

    test_transaction = {

        "transaction_id": "TXN-TEST-001",

        "customer_id": "CUST-TEST-001",

        "account_id": "ACC-TEST-001",

        "amount": 12500.00,

        "channel": "ONLINE",

        "country": "US",

        "is_fraud": 1,

        "fraud_type": "TEST_FRAUD",
    }

    inserted = write_prediction(
        transaction=test_transaction,
        fraud_probability=0.91,
        predicted_fraud=1,
        risk_level="CRITICAL",
    )

    print()

    if inserted:

        print(
            "Prediction written successfully."
        )

    else:

        print(
            "Duplicate prediction skipped."
        )

    print()

    print(
        f"Unique CSV transactions: "
        f"{get_csv_transaction_count()}"
    )

    print()

    print("=" * 70)

    print(
        f"Prediction file: "
        f"{PREDICTION_FILE}"
    )

    print("=" * 70)