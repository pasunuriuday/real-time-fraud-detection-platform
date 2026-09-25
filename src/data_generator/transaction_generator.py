import random
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from faker import Faker


# ---------------------------------------------------------
# INITIALIZE FAKER
# ---------------------------------------------------------

fake = Faker("en_US")


# ---------------------------------------------------------
# TRANSACTION CONFIGURATION
# ---------------------------------------------------------

CHANNELS = [
    "CARD",
    "ATM",
    "MOBILE",
    "ONLINE",
]

TRANSACTION_TYPES = [
    "PURCHASE",
    "WITHDRAWAL",
    "TRANSFER",
    "PAYMENT",
]

MERCHANT_CATEGORIES = [
    "GROCERY",
    "RESTAURANT",
    "FUEL",
    "ELECTRONICS",
    "TRAVEL",
    "ENTERTAINMENT",
    "HEALTHCARE",
    "RETAIL",
]


# ---------------------------------------------------------
# GENERATE RANDOM TIMESTAMP
# ---------------------------------------------------------

def generate_timestamp(days_back=90):
    """
    Generate a random UTC timestamp within the
    previous N days.
    """

    now = datetime.now(timezone.utc)

    random_seconds = random.randint(
        0,
        days_back * 24 * 60 * 60
    )

    timestamp = now - timedelta(
        seconds=random_seconds
    )

    return timestamp


# ---------------------------------------------------------
# GENERATE ONE TRANSACTION
# ---------------------------------------------------------

def generate_transaction(account_row):
    """
    Generate one synthetic banking transaction
    for an existing banking account.
    """

    # Select transaction channel
    channel = random.choice(CHANNELS)

    # Determine transaction type based on channel
    if channel == "ATM":

        transaction_type = "WITHDRAWAL"

    elif channel == "CARD":

        transaction_type = "PURCHASE"

    else:

        transaction_type = random.choice(
            [
                "PURCHASE",
                "TRANSFER",
                "PAYMENT",
            ]
        )

    # Generate transaction amount
    amount = round(
        random.uniform(5, 2500),
        2
    )

    # Build transaction record
    transaction = {

        # Transaction identifiers
        "transaction_id":
            f"TXN-{uuid.uuid4().hex[:12].upper()}",

        "customer_id":
            account_row["customer_id"],

        "account_id":
            account_row["account_id"],

        # Transaction details
        "timestamp":
            generate_timestamp().isoformat(),

        "amount":
            amount,

        "currency":
            "USD",

        "transaction_type":
            transaction_type,

        "channel":
            channel,

        # Merchant information
        "merchant_id":
            f"MER-{uuid.uuid4().hex[:8].upper()}",

        "merchant_name":
            fake.company(),

        "merchant_category":
            random.choice(
                MERCHANT_CATEGORIES
            ),

        # Device information
        "device_id":
            f"DEV-{uuid.uuid4().hex[:8].upper()}",

        "ip_address":
            fake.ipv4_public(),

        # Transaction location
        "city":
            fake.city(),

        "state":
            fake.state_abbr(),

        "country":
            "US",

        "latitude":
            float(fake.latitude()),

        "longitude":
            float(fake.longitude()),

        # Fraud fields
        # All transactions are normal for now.
        # Fraud scenarios will be added later.
        "is_fraud":
            0,

        "fraud_type":
            "NONE",
    }

    return transaction


# ---------------------------------------------------------
# GENERATE TRANSACTIONS FOR ALL ACCOUNTS
# ---------------------------------------------------------

def generate_transactions(accounts_df):
    """
    Generate between 10 and 30 transactions
    for every banking account.
    """

    transactions = []

    # Loop through every account
    for _, account in accounts_df.iterrows():

        # Each account gets 10-30 transactions
        number_of_transactions = random.randint(
            10,
            30
        )

        # Generate transactions
        for _ in range(number_of_transactions):

            transaction = generate_transaction(
                account
            )

            transactions.append(
                transaction
            )

    return transactions


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":

    # Input dataset
    accounts_path = (
        "data/sample/accounts.csv"
    )

    # Output dataset
    output_path = (
        "data/sample/transactions.csv"
    )

    print(
        "Loading banking accounts..."
    )

    # Read account dataset
    accounts_df = pd.read_csv(
        accounts_path
    )

    print(
        f"Loaded {len(accounts_df)} accounts"
    )

    print(
        "Generating transactions..."
    )

    # Generate transactions
    transactions = generate_transactions(
        accounts_df
    )

    # Convert Python list to Pandas DataFrame
    transactions_df = pd.DataFrame(
        transactions
    )

    # Sort transactions chronologically
    transactions_df = (
        transactions_df.sort_values(
            by="timestamp"
        )
    )

    # Save dataset
    transactions_df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print()
    print(
        f"Generated {len(transactions_df)} transactions"
    )

    print(
        f"Saved dataset to: {output_path}"
    )

    print()
    print("Sample transactions:")

    print(
        transactions_df.head()
    )

    # -----------------------------------------------------
    # BASIC DATA QUALITY CHECKS
    # -----------------------------------------------------

    print()
    print("Data Quality Checks:")

    # Duplicate transaction IDs
    duplicate_transactions = (
        transactions_df[
            "transaction_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate transaction IDs:",
        duplicate_transactions
    )

    # Missing account IDs
    missing_accounts = (
        transactions_df[
            "account_id"
        ]
        .isna()
        .sum()
    )

    print(
        "Missing account IDs:",
        missing_accounts
    )

    # Missing customer IDs
    missing_customers = (
        transactions_df[
            "customer_id"
        ]
        .isna()
        .sum()
    )

    print(
        "Missing customer IDs:",
        missing_customers
    )

    # -----------------------------------------------------
    # TRANSACTION CHANNEL DISTRIBUTION
    # -----------------------------------------------------

    print()
    print(
        "Transaction Channel Distribution:"
    )

    print(
        transactions_df[
            "channel"
        ]
        .value_counts()
    )

    # -----------------------------------------------------
    # TRANSACTION TYPE DISTRIBUTION
    # -----------------------------------------------------

    print()
    print(
        "Transaction Type Distribution:"
    )

    print(
        transactions_df[
            "transaction_type"
        ]
        .value_counts()
    )