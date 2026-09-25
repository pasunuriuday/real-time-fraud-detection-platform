from faker import Faker
import random
import uuid

import pandas as pd


fake = Faker("en_US")


def generate_customer():
    """Generate one synthetic banking customer."""

    customer = {
        "customer_id": f"CUST-{uuid.uuid4().hex[:8].upper()}",
        "name": fake.name(),
        "age": random.randint(18, 80),
        "email": fake.email(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "account_type": random.choice(
            ["CHECKING", "SAVINGS", "CREDIT"]
        ),
    }

    return customer


def generate_customers(number_of_customers):
    """Generate multiple synthetic banking customers."""

    customers = []

    for _ in range(number_of_customers):
        customers.append(generate_customer())

    return customers


if __name__ == "__main__":

    customers = generate_customers(1000)

    df = pd.DataFrame(customers)

    output_path = "data/sample/customers.csv"

    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} customers")
    print(f"Saved dataset to: {output_path}")

    print("\nSample customers:")
    print(df.head())