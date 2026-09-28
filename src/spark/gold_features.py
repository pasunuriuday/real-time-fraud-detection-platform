from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    lit,
    current_timestamp,
    hour,
    count,
    sum as spark_sum,
    avg,
    max as spark_max,
    round as spark_round,
)


# =========================================================
# CONFIGURATION
# =========================================================

SILVER_PATH = "data/silver/transactions"
GOLD_PATH = "data/gold/fraud_features"
GOLD_CUSTOMER_PATH = "data/gold/customer_risk_summary"


# =========================================================
# CREATE SPARK SESSION
# =========================================================

def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("FraudDetectionGoldLayer")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# =========================================================
# READ SILVER DATA
# =========================================================

def read_silver_data(spark):

    print()
    print("Reading Silver transaction data...")

    silver_df = spark.read.parquet(
        SILVER_PATH
    )

    print(
        f"Silver records loaded: {silver_df.count()}"
    )

    return silver_df


# =========================================================
# TRANSACTION FEATURE ENGINEERING
# =========================================================

def create_transaction_features(silver_df):

    print()
    print("Creating transaction-level fraud features...")

    features_df = (
        silver_df

        # -------------------------------------------------
        # TRANSACTION HOUR
        # -------------------------------------------------

        .withColumn(
            "transaction_hour",
            hour(
                col("transaction_timestamp")
            )
        )

        # -------------------------------------------------
        # LARGE TRANSACTION FEATURE
        # -------------------------------------------------

        .withColumn(
            "large_transaction_flag",
            when(
                col("amount") >= 5000,
                1
            ).otherwise(0)
        )

        # -------------------------------------------------
        # VERY LARGE TRANSACTION FEATURE
        # -------------------------------------------------

        .withColumn(
            "very_large_transaction_flag",
            when(
                col("amount") >= 10000,
                1
            ).otherwise(0)
        )

        # -------------------------------------------------
        # NIGHT TRANSACTION FEATURE
        # -------------------------------------------------

        .withColumn(
            "night_transaction_flag",
            when(
                (
                    col("transaction_hour") >= 0
                )
                &
                (
                    col("transaction_hour") < 5
                ),
                1
            ).otherwise(0)
        )

        # -------------------------------------------------
        # INTERNATIONAL TRANSACTION FEATURE
        # -------------------------------------------------

        .withColumn(
            "international_transaction_flag",
            when(
                col("country") != "US",
                1
            ).otherwise(0)
        )

        # -------------------------------------------------
        # EXISTING FRAUD-SIMULATION SIGNALS
        # -------------------------------------------------

        .withColumn(
            "new_device_risk",
            when(
                col("new_device_flag") == 1,
                1
            ).otherwise(0)
        )

        .withColumn(
            "high_amount_risk",
            when(
                col("high_amount_flag") == 1,
                1
            ).otherwise(0)
        )

        .withColumn(
            "geolocation_risk",
            when(
                col("geo_anomaly_flag") == 1,
                1
            ).otherwise(0)
        )

        .withColumn(
            "velocity_risk",
            when(
                col("velocity_flag") == 1,
                1
            ).otherwise(0)
        )
    )

    return features_df


# =========================================================
# CALCULATE COMBINED RISK SIGNAL
# =========================================================

def calculate_combined_risk(features_df):

    print()
    print("Calculating combined fraud-risk signals...")

    risk_df = (
        features_df

        .withColumn(
            "combined_risk_signal",

            col("new_device_risk")
            + col("high_amount_risk")
            + col("geolocation_risk")
            + col("velocity_risk")
            + col("large_transaction_flag")
            + col("very_large_transaction_flag")
            + col("night_transaction_flag")
            + col("international_transaction_flag")
        )
    )

    return risk_df


# =========================================================
# CALCULATE RISK SCORE
# =========================================================

def calculate_risk_score(risk_df):

    print()
    print("Calculating transaction risk scores...")

    scored_df = (
        risk_df

        .withColumn(
            "risk_score",

            (
                col("new_device_risk") * 20
                + col("high_amount_risk") * 25
                + col("geolocation_risk") * 25
                + col("velocity_risk") * 20
                + col("large_transaction_flag") * 10
                + col("very_large_transaction_flag") * 10
                + col("night_transaction_flag") * 5
                + col("international_transaction_flag") * 10
            )
        )

        # Keep score within 0-100.
        .withColumn(
            "risk_score",

            when(
                col("risk_score") > 100,
                100
            ).otherwise(
                col("risk_score")
            )
        )
    )

    return scored_df


# =========================================================
# ASSIGN RISK LEVEL
# =========================================================

def assign_risk_level(scored_df):

    print()
    print("Assigning transaction risk levels...")

    classified_df = (
        scored_df

        .withColumn(
            "risk_level",

            when(
                col("risk_score") >= 70,
                "CRITICAL"
            )

            .when(
                col("risk_score") >= 40,
                "HIGH"
            )

            .when(
                col("risk_score") >= 20,
                "MEDIUM"
            )

            .otherwise(
                "LOW"
            )
        )

        .withColumn(
            "gold_processing_timestamp",
            current_timestamp()
        )
    )

    return classified_df


# =========================================================
# CUSTOMER-LEVEL RISK AGGREGATION
# =========================================================

def create_customer_risk_summary(gold_df):

    print()
    print("Creating customer-level risk summary...")

    customer_df = (
        gold_df

        .groupBy(
            "customer_id"
        )

        .agg(

            count(
                "*"
            ).alias(
                "total_transactions"
            ),

            spark_round(
                spark_sum(
                    "amount"
                ),
                2
            ).alias(
                "total_transaction_amount"
            ),

            spark_round(
                avg(
                    "amount"
                ),
                2
            ).alias(
                "average_transaction_amount"
            ),

            spark_sum(
                "is_fraud"
            ).alias(
                "known_fraud_transactions"
            ),

            spark_round(
                avg(
                    "risk_score"
                ),
                2
            ).alias(
                "average_risk_score"
            ),

            spark_max(
                "risk_score"
            ).alias(
                "maximum_risk_score"
            ),

            spark_sum(
                "new_device_risk"
            ).alias(
                "new_device_events"
            ),

            spark_sum(
                "geolocation_risk"
            ).alias(
                "geolocation_anomalies"
            ),

            spark_sum(
                "velocity_risk"
            ).alias(
                "velocity_events"
            ),

            spark_sum(
                "international_transaction_flag"
            ).alias(
                "international_transactions"
            ),
        )

        .withColumn(
            "customer_risk_level",

            when(
                col("maximum_risk_score") >= 70,
                "CRITICAL"
            )

            .when(
                col("maximum_risk_score") >= 40,
                "HIGH"
            )

            .when(
                col("maximum_risk_score") >= 20,
                "MEDIUM"
            )

            .otherwise(
                "LOW"
            )
        )

        .withColumn(
            "gold_processing_timestamp",
            current_timestamp()
        )
    )

    return customer_df


# =========================================================
# PRINT GOLD REPORT
# =========================================================

def print_gold_report(gold_df):

    total_count = gold_df.count()

    fraud_count = (
        gold_df
        .filter(
            col("is_fraud") == 1
        )
        .count()
    )

    critical_count = (
        gold_df
        .filter(
            col("risk_level") == "CRITICAL"
        )
        .count()
    )

    high_count = (
        gold_df
        .filter(
            col("risk_level") == "HIGH"
        )
        .count()
    )

    medium_count = (
        gold_df
        .filter(
            col("risk_level") == "MEDIUM"
        )
        .count()
    )

    low_count = (
        gold_df
        .filter(
            col("risk_level") == "LOW"
        )
        .count()
    )

    print()
    print("=" * 70)
    print("GOLD FRAUD FEATURE REPORT")
    print("=" * 70)

    print(
        f"Total transactions    : {total_count}"
    )

    print(
        f"Known fraud           : {fraud_count}"
    )

    print(
        f"Critical risk         : {critical_count}"
    )

    print(
        f"High risk             : {high_count}"
    )

    print(
        f"Medium risk           : {medium_count}"
    )

    print(
        f"Low risk              : {low_count}"
    )

    print("=" * 70)


# =========================================================
# WRITE GOLD DATA
# =========================================================

def write_gold_data(
    gold_df,
    customer_df
):

    print()
    print(
        f"Writing transaction Gold data to: "
        f"{GOLD_PATH}"
    )

    (
        gold_df.write
        .mode("overwrite")
        .parquet(
            GOLD_PATH
        )
    )

    print(
        "Transaction Gold dataset saved."
    )

    print()
    print(
        f"Writing customer risk summary to: "
        f"{GOLD_CUSTOMER_PATH}"
    )

    (
        customer_df.write
        .mode("overwrite")
        .parquet(
            GOLD_CUSTOMER_PATH
        )
    )

    print(
        "Customer Gold dataset saved."
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "REAL-TIME FRAUD DETECTION - GOLD LAYER"
    )
    print("=" * 70)

    spark = create_spark_session()

    print()
    print(
        f"Spark version: {spark.version}"
    )

    # -----------------------------------------------------
    # READ SILVER
    # -----------------------------------------------------

    silver_df = read_silver_data(
        spark
    )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    gold_df = create_transaction_features(
        silver_df
    )

    # -----------------------------------------------------
    # COMBINED SIGNAL
    # -----------------------------------------------------

    gold_df = calculate_combined_risk(
        gold_df
    )

    # -----------------------------------------------------
    # RISK SCORE
    # -----------------------------------------------------

    gold_df = calculate_risk_score(
        gold_df
    )

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    gold_df = assign_risk_level(
        gold_df
    )

    # -----------------------------------------------------
    # CUSTOMER AGGREGATION
    # -----------------------------------------------------

    customer_df = create_customer_risk_summary(
        gold_df
    )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    print_gold_report(
        gold_df
    )

    # -----------------------------------------------------
    # SHOW TRANSACTION FEATURES
    # -----------------------------------------------------

    print()
    print("Sample Gold fraud features:")

    gold_df.select(
        "transaction_id",
        "customer_id",
        "amount",
        "channel",
        "country",
        "new_device_risk",
        "high_amount_risk",
        "geolocation_risk",
        "velocity_risk",
        "international_transaction_flag",
        "combined_risk_signal",
        "risk_score",
        "risk_level",
        "is_fraud",
    ).orderBy(
        col("risk_score").desc()
    ).show(
        10,
        truncate=False
    )

    # -----------------------------------------------------
    # SHOW CUSTOMER RISK
    # -----------------------------------------------------

    print()
    print("Sample high-risk customers:")

    customer_df.select(
        "customer_id",
        "total_transactions",
        "total_transaction_amount",
        "known_fraud_transactions",
        "average_risk_score",
        "maximum_risk_score",
        "customer_risk_level",
    ).orderBy(
        col("maximum_risk_score").desc()
    ).show(
        10,
        truncate=False
    )

    # -----------------------------------------------------
    # WRITE GOLD
    # -----------------------------------------------------

    write_gold_data(
        gold_df,
        customer_df
    )

    spark.stop()

    print()
    print("=" * 70)
    print(
        "GOLD PROCESSING COMPLETED"
    )
    print("=" * 70)