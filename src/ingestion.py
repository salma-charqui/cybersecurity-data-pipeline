"""
Ingestion of the CIC-IDS2017 dataset (DistriNet corrected version) - Bronze layer.

Loads raw CSV files, infers the schema, cleans column names,
and prepares data for the Bronze layer.

Local usage:
    python -m src.ingestion [--input data/raw/] [--output data/bronze/]

Databricks usage:
    from src.ingestion import ingest_csv_to_bronze_memory
    df = ingest_csv_to_bronze_memory(spark, "data/raw/")
"""

import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "cybersecurity-pipeline") -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )


def clean_column_name(name: str) -> str:
    """
    Clean a CIC-IDS2017 column name.
    Example: ' Source IP' -> 'source_ip'
             'Flow Duration' -> 'flow_duration'
             'Destination Port' -> 'destination_port'
    """
    return name.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def ingest_csv_to_bronze(
    spark: SparkSession,
    input_path: str,
    output_path: str,
) -> None:
    """
    Load raw CSV files (DistriNet corrected CIC-IDS2017) and save as Parquet (Bronze layer).

    Handles specific features of the corrected version:
    - Column names with spaces (e.g., ' Source IP' -> 'source_ip')
    - Infinite values replaced by null
    """
    input_dir = Path(input_path)
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        logger.warning("No CSV file found in %s" % input_path)
        logger.info("You can generate a synthetic dataset with:")
        logger.info("   generate_sample_dataset('data/raw/', 10000)")
        return

    logger.info("Found %d CSV file(s) to ingest" % len(csv_files))

    for csv_file in csv_files:
        logger.info("Ingesting %s ..." % csv_file.name)

        # Read CSV
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("nanValue", "")
            .option("nullValue", "NaN,Infinity,-Infinity,Inf,-Inf")
            .csv(str(csv_file))
        )

        # Clean column names
        for old_name in df.columns:
            new_name = clean_column_name(old_name)
            if old_name != new_name:
                df = df.withColumnRenamed(old_name, new_name)

        # Add ingestion metadata
        df = (
            df
            .withColumn("_source_file", F.lit(csv_file.name))
            .withColumn("_ingestion_ts", F.current_timestamp())
        )

        # Stats
        row_count = df.count()
        col_count = len(df.columns)
        logger.info("  -> %d rows, %d columns" % (row_count, col_count))

        # Save as Parquet
        output_file = output_path + "/" + csv_file.stem
        (
            df.write
            .mode("overwrite")
            .parquet(output_file)
        )
        logger.info("  -> Saved to %s" % output_file)

    logger.info("Bronze ingestion completed")


def ingest_csv_to_bronze_memory(
    spark: SparkSession,
    input_path: str,
):
    """
    In-memory version of Bronze ingestion (for testing on Databricks Serverless).

    Reads the CSV, applies column cleaning, adds metadata,
    but DOES NOT persist as Parquet (returns the DataFrame instead).

    Usage:
        df = ingest_csv_to_bronze_memory(spark, "data/raw/")
        df.show()
    """
    input_dir = Path(input_path)
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        logger.warning("No CSV file found in %s" % input_path)
        return None

    logger.info("Found %d CSV file(s) to ingest" % len(csv_files))

    # For now, only one CSV is handled at a time
    csv_file = csv_files[0]
    logger.info("Ingesting %s ..." % csv_file.name)

    # Read CSV
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("nanValue", "")
        .option("nullValue", "NaN,Infinity,-Infinity,Inf,-Inf")
        .csv(str(csv_file))
    )

    # Clean column names
    for old_name in df.columns:
        new_name = clean_column_name(old_name)
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)

    # Add ingestion metadata
    df = (
        df
        .withColumn("_source_file", F.lit(csv_file.name))
        .withColumn("_ingestion_ts", F.current_timestamp())
    )

    # Stats
    row_count = df.count()
    col_count = len(df.columns)
    logger.info("  -> %d rows, %d columns" % (row_count, col_count))
    logger.info("Bronze ingestion completed (in memory)")

    return df


def generate_sample_dataset(output_path: str, n_rows: int = 10000) -> None:
    """
    Generate a synthetic dataset to test the pipeline
    without downloading the full CIC-IDS2017 dataset (several GB).
    """
    import random
    import pandas as pd
    from datetime import datetime, timedelta

    random.seed(42)
    labels = ["BENIGN", "DoS-Hulk", "DoS-GoldenEye", "PortScan", "DDoS", "Brute Force"]
    protocols = ["TCP", "UDP", "ICMP"]
    sources = [f"192.168.1.{i}" for i in range(1, 11)] + [f"10.0.0.{i}" for i in range(1, 6)]
    destinations = ["8.8.8.8", "1.1.1.1", "8.8.4.4", "9.9.9.9"]

    rows = []
    for _ in range(n_rows):
        label = random.choices(labels, weights=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])[0]
        rows.append({
            "source_ip": random.choice(sources),
            "destination_ip": random.choice(destinations),
            "protocol": random.choice(protocols),
            "port": random.choice([22, 80, 443, 53, 8080, 25, 3389]),
            "bytes": random.randint(100, 100000),
            "label": label,
            "timestamp": (datetime(2024, 1, 1) + timedelta(seconds=random.randint(0, 86400*7))).isoformat()
        })

    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) / "cic_sample.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    logger.info("Synthetic dataset created: %s (%d rows)" % (output_file, n_rows))

def ingest_parquet_to_bronze_memory(
    spark: SparkSession,
    file_path: str,
):
    """
    In-memory version of Bronze ingestion for Parquet files (DistriNet version).
    
    Reads a Parquet file, cleans column names, adds ingestion metadata,
    but DOES NOT persist (returns the DataFrame).
    
    Usage:
        df = ingest_parquet_to_bronze_memory(spark, "/Volumes/.../file.parquet")
    """
    from pathlib import Path
    
    file_path_obj = Path(file_path)
    logger.info("Ingesting Parquet file: %s" % file_path_obj.name)
    
    # Read Parquet (already typed, no need for inferSchema)
    df = spark.read.parquet(file_path)
    
    # Clean column names (same logic as CSV)
    for old_name in df.columns:
        new_name = clean_column_name(old_name)
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)
    
    # Add ingestion metadata
    df = (
        df
        .withColumn("_source_file", F.lit(file_path_obj.name))
        .withColumn("_ingestion_ts", F.current_timestamp())
    )
    
    # Stats
    row_count = df.count()
    col_count = len(df.columns)
    logger.info("  -> %d rows, %d columns" % (row_count, col_count))
    logger.info("Bronze ingestion completed (in memory, from Parquet)")
    
    return df

# === For local CLI usage ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CIC-IDS2017 ingestion -> Bronze layer")
    parser.add_argument("--input", default="data/raw/", help="Raw CSV folder")
    parser.add_argument("--output", default="data/bronze/", help="Parquet output folder")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a synthetic dataset")
    parser.add_argument("--sample-rows", type=int, default=10000)
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset(args.input, args.sample_rows)
    else:
        spark = create_spark_session()
        try:
            ingest_csv_to_bronze(spark, args.input, args.output)
        finally:
            spark.stop()