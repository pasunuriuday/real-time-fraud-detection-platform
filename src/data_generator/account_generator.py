import random
import uuid

import pandas as pd


ACCOUNT_TYPES = ["CHECKING", "SAVINGS", "CREDIT"]


def generate_account(customer_id):
    """Generate one synthetic banking account for a customer."""

    account_type = random.choice(ACCOUNT_TYPES)

    account = {
        "account_id": f"ACC-{uuid.uuid4().hex[:10].upper()}",
        "customer_id": customer_id,
        "account_type": account_type,
        "account_status": random.choice(
            ["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "DORMANT"]
        ),
        "balance": round(random.uniform(100, 50000), 2),
        "credit_limit": (
            round(random.uniform(1000, 25000), 2)
            if account_type == "CREDIT"
            else 0.0
        ),
    }

    return account


def generate_accounts(customers_df):
    """Generate 1 to 3 accounts for every customer."""

    accounts = []

    for customer_id in customers_df["customer_id"]:

        number_of_accounts = random.randint(1, 3)

        for _ in range(number_of_accounts):
            accounts.append(generate_account(customer_id))

    return accounts


if __name__ == "__main__":

    customer_path = "data/sample/customers.csv"
    output_path = "data/sample/accounts.csv"

    customers_df = pd.read_csv(customer_path)

    accounts = generate_accounts(customers_df)

    accounts_df = pd.DataFrame(accounts)

    accounts_df.to_csv(output_path, index=False)

    print(f"Loaded {len(customers_df)} customers")
    print(f"Generated {len(accounts_df)} accounts")
    print(f"Saved dataset to: {output_path}")

    print("\nSample accounts:")
    print(accounts_df.head())