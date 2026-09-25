import json
from confluent_kafka import Consumer, KafkaError


# =========================================================
# KAFKA CONFIGURATION
# =========================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "banking-transactions"

# Name of our consumer group
CONSUMER_GROUP = "fraud-detection-consumer"

# For this test, read only 10 messages
MAX_MESSAGES = 10


# =========================================================
# CREATE KAFKA CONSUMER
# =========================================================

def create_consumer():
    """
    Create and configure the Kafka consumer.
    """

    consumer_config = {

        # Kafka broker location
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,

        # Consumer group name
        "group.id": CONSUMER_GROUP,

        # Start from earliest message if this group
        # does not already have a saved offset.
        "auto.offset.reset": "earliest",

        # Automatically save processed offsets
        "enable.auto.commit": True,
    }

    return Consumer(consumer_config)


# =========================================================
# DISPLAY TRANSACTION
# =========================================================

def display_transaction(transaction, message):
    """
    Display important transaction and Kafka metadata.
    """

    print("=" * 70)

    print(
        f"Transaction ID : "
        f"{transaction.get('transaction_id')}"
    )

    print(
        f"Customer ID    : "
        f"{transaction.get('customer_id')}"
    )

    print(
        f"Account ID     : "
        f"{transaction.get('account_id')}"
    )

    print(
        f"Amount         : "
        f"${transaction.get('amount')}"
    )

    print(
        f"Channel        : "
        f"{transaction.get('channel')}"
    )

    print(
        f"Country        : "
        f"{transaction.get('country')}"
    )

    print(
        f"Fraud          : "
        f"{transaction.get('is_fraud')}"
    )

    print(
        f"Fraud Type     : "
        f"{transaction.get('fraud_type')}"
    )

    print()
    print("Kafka Metadata")

    print(
        f"Partition      : "
        f"{message.partition()}"
    )

    print(
        f"Offset         : "
        f"{message.offset()}"
    )

    print("=" * 70)
    print()


# =========================================================
# CONSUME TRANSACTIONS
# =========================================================

def consume_transactions(consumer):
    """
    Read transactions from the Kafka topic.
    """

    print()
    print(f"Kafka broker : {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka topic  : {KAFKA_TOPIC}")
    print(f"Consumer group: {CONSUMER_GROUP}")

    print()
    print("Waiting for transactions...")
    print()

    # Subscribe to our banking transaction topic
    consumer.subscribe(
        [KAFKA_TOPIC]
    )

    processed_messages = 0

    try:

        while processed_messages < MAX_MESSAGES:

            # Ask Kafka for a message.
            #
            # timeout=1 means wait up to one second.
            message = consumer.poll(
                timeout=1.0
            )

            # No message available yet
            if message is None:
                continue

            # Kafka returned an error
            if message.error():

                if (
                    message.error().code()
                    == KafkaError._PARTITION_EOF
                ):
                    continue

                print(
                    f"Kafka error: "
                    f"{message.error()}"
                )

                continue

            # ---------------------------------------------
            # DECODE JSON
            # ---------------------------------------------

            try:

                transaction = json.loads(
                    message.value().decode("utf-8")
                )

            except json.JSONDecodeError as error:

                print(
                    f"Invalid JSON message: {error}"
                )

                continue

            # ---------------------------------------------
            # DISPLAY TRANSACTION
            # ---------------------------------------------

            processed_messages += 1

            print(
                f"Message "
                f"{processed_messages}/{MAX_MESSAGES}"
            )

            display_transaction(
                transaction,
                message
            )

    except KeyboardInterrupt:

        print()
        print(
            "Consumer stopped by user."
        )

    finally:

        # Important:
        # close() commits offsets and leaves
        # the consumer group cleanly.
        consumer.close()

        print()
        print(
            "Kafka consumer closed."
        )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "REAL-TIME FRAUD DETECTION - KAFKA CONSUMER"
    )
    print("=" * 70)

    consumer = create_consumer()

    consume_transactions(
        consumer
    )

    print("=" * 70)
    print(
        "KAFKA CONSUMER FINISHED"
    )
    print("=" * 70)