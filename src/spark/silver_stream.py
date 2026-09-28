from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    current_timestamp,
    to_timestamp,
    when,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)


# =========================================================
# CONFIGURATION
# =========================================================

BRONZE_PATH = "data/bronze/transactions"
SILVER_PATH = "data/silver/transactions"


# =========================================================
# TRANSACTION JSON SCHEMA
# =========================================================

TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("account_id", StringType(), True),
        StructField("timestamp", StringType(), True),

        StructField("amount", DoubleType(), True),
        StructField("currency", StringType(), True),

        StructField("transaction_type", StringType(), True),
        StructField("channel", StringType(), True),

        StructField("merchant_id", StringType(), True),
        StructField("merchant_name", StringType(), True),
        StructField("merchant_category", StringType(), True),

        StructField("device_id", StringType(), True),
        StructField("ip_address", StringType(), True),

        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("country", StringType(), True),

        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),

        StructField("is_fraud", IntegerType(), True),
        StructField("fraud_type", StringType(), True),

        StructField("new_device_flag", IntegerType(), True),
        StructField("high_amount_flag", IntegerType(), True),
        StructField("geo_anomaly_flag", IntegerType(), True),
        StructField("velocity_flag", IntegerType(), True),
    ]
)


# =========================================================
# CREATE SPARK SESSION
# =========================================================

def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("FraudDetectionSilverLayer")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# =========================================================
# READ BRONZE DATA
# =========================================================

def read_bronze_data(spark):

    print()
    print("Reading Bronze transaction data...")

    bronze_df = spark.read.parquet(
        BRONZE_PATH
    )

    print(
        f"Bronze records loaded: {bronze_df.count()}"
    )

    return bronze_df


# =========================================================
# PARSE RAW JSON
# =========================================================

def parse_transactions(bronze_df):

    print()
    print("Parsing transaction JSON...")

    parsed_df = bronze_df.withColumn(
        "transaction",
        from_json(
            col("raw_json"),
            TRANSACTION_SCHEMA
        )
    )

    silver_df = parsed_df.select(

        # Transaction fields
        col("transaction.transaction_id")
        .alias("transaction_id"),

        col("transaction.customer_id")
        .alias("customer_id"),

        col("transaction.account_id")
        .alias("account_id"),

        to_timestamp(
            col("transaction.timestamp")
        ).alias("transaction_timestamp"),

        col("transaction.amount")
        .alias("amount"),

        col("transaction.currency")
        .alias("currency"),

        col("transaction.transaction_type")
        .alias("transaction_type"),

        col("transaction.channel")
        .alias("channel"),

        col("transaction.merchant_id")
        .alias("merchant_id"),

        col("transaction.merchant_name")
        .alias("merchant_name"),

        col("transaction.merchant_category")
        .alias("merchant_category"),

        col("transaction.device_id")
        .alias("device_id"),

        col("transaction.ip_address")
        .alias("ip_address"),

        col("transaction.city")
        .alias("city"),

        col("transaction.state")
        .alias("state"),

        col("transaction.country")
        .alias("country"),

        col("transaction.latitude")
        .alias("latitude"),

        col("transaction.longitude")
        .alias("longitude"),

        col("transaction.is_fraud")
        .alias("is_fraud"),

        col("transaction.fraud_type")
        .alias("fraud_type"),

        col("transaction.new_device_flag")
        .alias("new_device_flag"),

        col("transaction.high_amount_flag")
        .alias("high_amount_flag"),

        col("transaction.geo_anomaly_flag")
        .alias("geo_anomaly_flag"),

        col("transaction.velocity_flag")
        .alias("velocity_flag"),

        # Preserve Kafka lineage
        col("kafka_partition"),

        col("kafka_offset"),

        col("kafka_timestamp"),

        col("bronze_ingestion_timestamp"),
    )

    return silver_df


# =========================================================
# DATA QUALITY VALIDATION
# =========================================================

def apply_data_quality(silver_df):

    print()
    print("Applying data-quality rules...")

    validated_df = (
        silver_df

        # Required identifiers must exist.
        .filter(
            col("transaction_id").isNotNull()
        )

        .filter(
            col("customer_id").isNotNull()
        )

        .filter(
            col("account_id").isNotNull()
        )

        # Transaction amount must be positive.
        .filter(
            col("amount").isNotNull()
        )

        .filter(
            col("amount") > 0
        )

        # Fraud label should be 0 or 1.
        .filter(
            col("is_fraud").isin(0, 1)
        )
    )

    return validated_df


# =========================================================
# ADD DATA-QUALITY STATUS
# =========================================================

def add_quality_columns(silver_df):

    quality_df = (
        silver_df

        .withColumn(
            "data_quality_status",

            when(
                col("transaction_timestamp").isNull(),
                "INVALID_TIMESTAMP"
            )

            .when(
                col("currency").isNull(),
                "MISSING_CURRENCY"
            )

            .when(
                col("channel").isNull(),
                "MISSING_CHANNEL"
            )

            .otherwise(
                "VALID"
            )
        )

        .withColumn(
            "silver_processing_timestamp",
            current_timestamp()
        )
    )

    return quality_df


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(silver_df):

    print()
    print("Removing duplicate transactions...")

    deduplicated_df = (
        silver_df
        .dropDuplicates(
            ["transaction_id"]
        )
    )

    return deduplicated_df


# =========================================================
# DATA QUALITY REPORT
# =========================================================

def print_quality_report(
    bronze_count,
    silver_df
):

    silver_count = silver_df.count()

    fraud_count = (
        silver_df
        .filter(
            col("is_fraud") == 1
        )
        .count()
    )

    normal_count = (
        silver_df
        .filter(
            col("is_fraud") == 0
        )
        .count()
    )

    print()
    print("=" * 70)
    print("SILVER DATA QUALITY REPORT")
    print("=" * 70)

    print(
        f"Bronze records       : {bronze_count}"
    )

    print(
        f"Silver records       : {silver_count}"
    )

    print(
        f"Rejected/Duplicate   : "
        f"{bronze_count - silver_count}"
    )

    print(
        f"Normal transactions  : {normal_count}"
    )

    print(
        f"Fraud transactions   : {fraud_count}"
    )

    if silver_count > 0:

        fraud_rate = (
            fraud_count
            / silver_count
            * 100
        )

        print(
            f"Fraud rate           : "
            f"{fraud_rate:.2f}%"
        )

    print("=" * 70)


# =========================================================
# WRITE SILVER DATA
# =========================================================

def write_silver_data(silver_df):

    print()
    print(
        f"Writing Silver data to: "
        f"{SILVER_PATH}"
    )

    (
        silver_df.write
        .mode("overwrite")
        .parquet(
            SILVER_PATH
        )
    )

    print(
        "Silver dataset saved successfully."
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "REAL-TIME FRAUD DETECTION - SILVER LAYER"
    )
    print("=" * 70)

    spark = create_spark_session()

    print()
    print(
        f"Spark version: {spark.version}"
    )

    # -----------------------------------------------------
    # READ BRONZE
    # -----------------------------------------------------

    bronze_df = read_bronze_data(
        spark
    )

    bronze_count = bronze_df.count()

    # -----------------------------------------------------
    # PARSE JSON
    # -----------------------------------------------------

    silver_df = parse_transactions(
        bronze_df
    )

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    silver_df = apply_data_quality(
        silver_df
    )

    # -----------------------------------------------------
    # QUALITY COLUMNS
    # -----------------------------------------------------

    silver_df = add_quality_columns(
        silver_df
    )

    # -----------------------------------------------------
    # DEDUPLICATE
    # -----------------------------------------------------

    silver_df = remove_duplicates(
        silver_df
    )

    # -----------------------------------------------------
    # SHOW SCHEMA
    # -----------------------------------------------------

    print()
    print("Silver schema:")

    silver_df.printSchema()

    # -----------------------------------------------------
    # QUALITY REPORT
    # -----------------------------------------------------

    print_quality_report(
        bronze_count,
        silver_df
    )

    # -----------------------------------------------------
    # SHOW SAMPLE
    # -----------------------------------------------------

    print()
    print("Sample Silver transactions:")

    silver_df.select(
        "transaction_id",
        "customer_id",
        "account_id",
        "transaction_timestamp",
        "amount",
        "channel",
        "country",
        "is_fraud",
        "fraud_type",
        "data_quality_status",
    ).show(
        10,
        truncate=False
    )

    # -----------------------------------------------------
    # WRITE SILVER
    # -----------------------------------------------------

    write_silver_data(
        silver_df
    )

    spark.stop()

    print()
    print("=" * 70)
    print(
        "SILVER PROCESSING COMPLETED"
    )
    print("=" * 70)