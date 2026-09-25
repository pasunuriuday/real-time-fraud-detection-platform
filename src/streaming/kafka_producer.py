import json
import time

import pandas as pd
from confluent_kafka import Producer


# =========================================================
# CONFIGURATION
# =========================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "banking-transactions"

INPUT_FILE = "data/sample/transactions_with_fraud.csv"

# Only send 100 transactions while testing.
TEST_RECORD_LIMIT = 100

# Small delay between events so we can simulate streaming.
DELAY_SECONDS = 0.05


# =========================================================
# DELIVERY CALLBACK
# =========================================================

def delivery_report(err, msg):
    """
    Kafka calls this function after attempting
    to deliver a message.
    """

    if err is not None:
        print(f"Message delivery failed: {err}")


# =========================================================
# CREATE KAFKA PRODUCER
# =========================================================

def create_producer():
    """
    Create the Kafka producer that connects
    to our local Kafka broker.
    """

    producer_config = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,

        # Wait for acknowledgement from Kafka.
        "acks": "all",

        # Prevent duplicate messages during retries.
        "enable.idempotence": True,
    }

    producer = Producer(producer_config)

    return producer


# =========================================================
# CLEAN PANDAS VALUES
# =========================================================

def clean_record(record):
    """
    Convert Pandas/NumPy values into standard
    Python values so json.dumps() can serialize them.
    """

    cleaned_record = {}

    for key, value in record.items():

        # Convert missing Pandas values to JSON null.
        if pd.isna(value):

            cleaned_record[key] = None

        # Convert NumPy values to standard Python values.
        elif hasattr(value, "item"):

            cleaned_record[key] = value.item()

        else:

            cleaned_record[key] = value

    return cleaned_record


# =========================================================
# STREAM TRANSACTIONS TO KAFKA
# =========================================================

def stream_transactions(
    producer,
    transactions_df,
    delay_seconds=DELAY_SECONDS
):
    """
    Send transaction records to Kafka one at a time.
    """

    total_records = len(transactions_df)

    sent_records = 0

    print()
    print(f"Streaming {total_records} transactions...")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print(f"Kafka broker: {KAFKA_BOOTSTRAP_SERVERS}")
    print()

    # -----------------------------------------------------
    # LOOP THROUGH TRANSACTIONS
    # -----------------------------------------------------

    for _, row in transactions_df.iterrows():

        # Convert Pandas row to Python dictionary.
        transaction = row.to_dict()

        # Clean Pandas/NumPy data types.
        transaction = clean_record(transaction)

        # Convert dictionary to JSON.
        message_value = json.dumps(transaction)

        # Use account_id as Kafka message key.
        #
        # This helps transactions belonging to the
        # same account stay on the same Kafka partition.
        message_key = str(
            transaction["account_id"]
        )

        try:

            # Send the event to Kafka.
            producer.produce(
                topic=KAFKA_TOPIC,
                key=message_key,
                value=message_value,
                callback=delivery_report,
            )

        except BufferError:

            # If the producer's local queue becomes full,
            # process outstanding Kafka events first.
            print(
                "Kafka producer queue is full. "
                "Waiting for messages to be delivered..."
            )

            producer.poll(1)

            producer.produce(
                topic=KAFKA_TOPIC,
                key=message_key,
                value=message_value,
                callback=delivery_report,
            )

        # Process Kafka delivery events.
        producer.poll(0)

        sent_records += 1

        # For our 100-record test, print every 10 records.
        if sent_records % 10 == 0:

            print(
                f"Sent {sent_records}/{total_records} transactions"
            )

        # Simulate transactions arriving over time.
        if delay_seconds > 0:

            time.sleep(delay_seconds)


    # =====================================================
    # WAIT FOR ALL MESSAGES TO FINISH
    # =====================================================

    print()
    print("Waiting for Kafka acknowledgements...")

    remaining_messages = producer.flush(10)

    print()

    if remaining_messages == 0:

        print(
            "All Kafka messages delivered successfully."
        )

    else:

        print(
            f"WARNING: {remaining_messages} "
            "messages were not delivered."
        )

    print(
        f"Total transactions processed: {sent_records}"
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("REAL-TIME FRAUD DETECTION - KAFKA PRODUCER")
    print("=" * 60)

    print()
    print("Loading fraud transaction dataset...")

    # -----------------------------------------------------
    # LOAD TRANSACTIONS
    # -----------------------------------------------------

    transactions_df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Total transactions available: "
        f"{len(transactions_df)}"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Only take the first 100 transactions for testing.
    # -----------------------------------------------------

    transactions_df = transactions_df.head(
        TEST_RECORD_LIMIT
    )

    print(
        f"Transactions selected for test: "
        f"{len(transactions_df)}"
    )

    # -----------------------------------------------------
    # CREATE PRODUCER
    # -----------------------------------------------------

    print()
    print("Connecting to Kafka...")

    producer = create_producer()

    # -----------------------------------------------------
    # STREAM DATA
    # -----------------------------------------------------

    stream_transactions(
        producer=producer,
        transactions_df=transactions_df,
        delay_seconds=DELAY_SECONDS,
    )

    print()
    print("=" * 60)
    print("KAFKA PRODUCER FINISHED")
    print("=" * 60)