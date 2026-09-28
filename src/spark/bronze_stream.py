from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
)


# =========================================================
# CONFIGURATION
# =========================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "banking-transactions"

BRONZE_PATH = "data/bronze/transactions"
CHECKPOINT_PATH = "data/checkpoints/bronze_transactions"


# =========================================================
# CREATE SPARK SESSION
# =========================================================

def create_spark_session():
    """
    Create the SparkSession used by our
    Structured Streaming application.
    """

    spark = (
        SparkSession.builder
        .appName("FraudDetectionBronzeStream")
        .master("local[*]")
        .getOrCreate()
    )

    # Reduce unnecessary console output.
    spark.sparkContext.setLogLevel("WARN")

    return spark


# =========================================================
# READ FROM KAFKA
# =========================================================

def read_kafka_stream(spark):
    """
    Create a streaming DataFrame connected
    to the banking-transactions Kafka topic.
    """

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_BOOTSTRAP_SERVERS
        )
        .option(
            "subscribe",
            KAFKA_TOPIC
        )
        .option(
            "startingOffsets",
            "earliest"
        )
        .load()
    )

    return kafka_df


# =========================================================
# CREATE BRONZE RECORDS
# =========================================================

def create_bronze_stream(kafka_df):
    """
    Preserve the raw Kafka event and important
    Kafka metadata in our Bronze layer.
    """

    bronze_df = (
        kafka_df.select(
            col("key")
            .cast("string")
            .alias("kafka_key"),

            col("value")
            .cast("string")
            .alias("raw_json"),

            col("topic")
            .alias("kafka_topic"),

            col("partition")
            .alias("kafka_partition"),

            col("offset")
            .alias("kafka_offset"),

            col("timestamp")
            .alias("kafka_timestamp"),
        )
        .withColumn(
            "bronze_ingestion_timestamp",
            current_timestamp()
        )
    )

    return bronze_df


# =========================================================
# WRITE BRONZE DATA
# =========================================================

def write_bronze_stream(bronze_df):
    """
    Write raw Kafka events to the Bronze layer.

    Parquet is used locally for now.
    Later, this architecture can be moved to
    Delta Lake / ADLS in Azure.
    """

    query = (
        bronze_df.writeStream
        .format("parquet")
        .outputMode("append")
        .option(
            "path",
            BRONZE_PATH
        )
        .option(
            "checkpointLocation",
            CHECKPOINT_PATH
        )
        .trigger(
            processingTime="5 seconds"
        )
        .start()
    )

    return query


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "REAL-TIME FRAUD DETECTION - BRONZE STREAM"
    )
    print("=" * 70)

    print()
    print(
        f"Kafka broker : {KAFKA_BOOTSTRAP_SERVERS}"
    )

    print(
        f"Kafka topic  : {KAFKA_TOPIC}"
    )

    print(
        f"Bronze path  : {BRONZE_PATH}"
    )

    print(
        f"Checkpoint   : {CHECKPOINT_PATH}"
    )

    print()

    # -----------------------------------------------------
    # START SPARK
    # -----------------------------------------------------

    spark = create_spark_session()

    print(
        f"Spark version: {spark.version}"
    )

    # -----------------------------------------------------
    # CONNECT TO KAFKA
    # -----------------------------------------------------

    print()
    print(
        "Connecting Spark Structured Streaming to Kafka..."
    )

    kafka_df = read_kafka_stream(
        spark
    )

    # -----------------------------------------------------
    # CREATE BRONZE DATAFRAME
    # -----------------------------------------------------

    bronze_df = create_bronze_stream(
        kafka_df
    )

    print()
    print("Bronze schema:")

    bronze_df.printSchema()

    # -----------------------------------------------------
    # START STREAM
    # -----------------------------------------------------

    query = write_bronze_stream(
        bronze_df
    )

    print()
    print(
        "Bronze streaming pipeline started."
    )

    print(
        "Waiting for Kafka transactions..."
    )

    print()
    print(
        "Press Control + C to stop the stream."
    )

    # Keep application running.
    try:

        query.awaitTermination()

    except KeyboardInterrupt:

        print()
        print(
            "Stopping Bronze streaming pipeline..."
        )

        query.stop()

        spark.stop()

        print(
            "Bronze streaming pipeline stopped."
        )