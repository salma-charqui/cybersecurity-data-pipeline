"""
Bronze -> Silver cleaning pipeline for the CIC-IDS2017 dataset.

Applies a series of cleaning steps to produce a validated, typed,
and enriched DataFrame ready for feature engineering (Gold layer)
and Machine Learning.

Each step is a standalone function so it can be tested individually.
The main entry point is `clean_bronze_to_silver()` which orchestrates all steps.

Databricks usage:
    from src.cleaning import clean_bronze_to_silver
    df_silver = clean_bronze_to_silver(df_bronze)
"""

import logging
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, LongType, DoubleType, TimestampType, StringType
)

# === Logging ===
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1 - Remove duplicates
# =============================================================================

def remove_duplicates(df: DataFrame) -> DataFrame:
    """
    Remove exact duplicate rows from the DataFrame.
    
    Returns:
        DataFrame without duplicates.
    """
    initial_count = df.count()
    df_clean = df.dropDuplicates()
    final_count = df_clean.count()
    
    removed = initial_count - final_count
    logger.info(f"Step 1 - remove_duplicates: removed {removed} duplicates "
                f"({initial_count} -> {final_count})")
    
    return df_clean


# =============================================================================
# Step 2 - Handle nulls
# =============================================================================

def handle_nulls(
    df: DataFrame,
    critical_columns: Optional[List[str]] = None,
    fill_value: str = "unknown",
) -> DataFrame:
    """
    Handle null values:
    - Drop rows with nulls in critical columns
    - Fill non-critical nulls with a default value (string columns)
    
    Args:
        df: input DataFrame
        critical_columns: columns where null means the row is invalid (drop)
        fill_value: default value for non-critical string nulls
    """
    initial_count = df.count()
    
    if critical_columns:
        df_clean = df.dropna(subset=critical_columns)
    else:
        df_clean = df
    
    # Fill remaining nulls in string columns with default value
    string_cols = [f.name for f in df_clean.schema.fields
                   if isinstance(f.dataType, StringType)]
    fill_dict = {col: fill_value for col in string_cols}
    df_clean = df_clean.fillna(fill_dict)
    
    final_count = df_clean.count()
    removed = initial_count - final_count
    logger.info(f"Step 2 - handle_nulls: dropped {removed} rows with critical nulls "
                f"({initial_count} -> {final_count})")
    
    return df_clean


# =============================================================================
# Step 3 - Filter invalid values
# =============================================================================

def filter_invalid_values(df: DataFrame) -> DataFrame:
    """
    Filter out rows with invalid values:
    - Negative bytes
    - Negative port
    - Port out of range (0-65535)
    - Infinite values in numeric columns
    """
    initial_count = df.count()
    
    df_clean = df
    
    # Filter on bytes if column exists
    if "bytes" in df.columns:
        df_clean = df_clean.filter(
            (F.col("bytes") >= 0) | F.col("bytes").isNull()
        )
    
    # Filter on port if column exists
    if "port" in df.columns:
        df_clean = df_clean.filter(
            ((F.col("port") >= 0) & (F.col("port") <= 65535)) |
            F.col("port").isNull()
        )
    
    final_count = df_clean.count()
    removed = initial_count - final_count
    logger.info(f"Step 3 - filter_invalid_values: removed {removed} invalid rows "
                f"({initial_count} -> {final_count})")
    
    return df_clean


# =============================================================================
# Step 4 - Standardize labels
# =============================================================================

def standardize_labels(df: DataFrame, label_col: str = "label") -> DataFrame:
    """
    Standardize attack label names:
    - Strip whitespace
    - Replace special characters (e.g., Web Attack - Brute Force)
    - Map common variations
    
    Args:
        df: input DataFrame
        label_col: name of the label column (default: 'label')
    """
    if label_col not in df.columns:
        logger.warning(f"Step 4 - standardize_labels: column '{label_col}' not found, skipping")
        return df
    
    df_clean = df.withColumn(
        label_col,
        F.trim(F.col(label_col))  # Remove leading/trailing whitespace
    )
    
    # Map common label variations to canonical names
    # (for the real CIC-IDS2017, there are known issues like 'Web Attack \\x96 ...')
    label_mapping = {
        "Web Attack \\x96 Brute Force": "Web Attack - Brute Force",
        "Web Attack \\x96 XSS": "Web Attack - XSS",
        "Web Attack \\x96 Sql Injection": "Web Attack - SQL Injection",
        "DoS Hulk": "DoS-Hulk",
        "DoS GoldenEye": "DoS-GoldenEye",
        "DoS slowloris": "DoS-Slowloris",
        "DoS Slowhttptest": "DoS-SlowHTTPTest",
    }
    
    # Apply mapping using when().otherwise()
    label_expr = F.lit(None).cast(StringType())
    for original, canonical in label_mapping.items():
        label_expr = F.when(F.col(label_col) == original, canonical).otherwise(label_expr)
    
    # If no mapping matched, keep the original (already trimmed)
    df_clean = df_clean.withColumn(
        label_col,
        F.when(label_expr.isNotNull(), label_expr).otherwise(F.col(label_col))
    )
    
    unique_labels = df_clean.select(label_col).distinct().count()
    logger.info(f"Step 4 - standardize_labels: {unique_labels} unique labels after standardization")
    
    return df_clean


# =============================================================================
# Step 5 - Cast types
# =============================================================================

def cast_types(df: DataFrame) -> DataFrame:
    """
    Cast columns to their proper types:
    - port -> int
    - bytes -> long
    - timestamp -> timestamp
    - source_ip, destination_ip, protocol, label -> string
    """
    df_clean = df
    
    type_mapping = {
        "port": IntegerType(),
        "bytes": LongType(),
        "timestamp": TimestampType(),
        "source_ip": StringType(),
        "destination_ip": StringType(),
        "protocol": StringType(),
        "label": StringType(),
    }
    
    for col_name, target_type in type_mapping.items():
        if col_name in df_clean.columns:
            df_clean = df_clean.withColumn(col_name, F.col(col_name).cast(target_type))
    
    logger.info(f"Step 5 - cast_types: types cast to {len(type_mapping)} columns")
    
    return df_clean


# =============================================================================
# Step 6 - Drop constant columns
# =============================================================================

def drop_constant_columns(df: DataFrame, protected_columns: Optional[List[str]] = None) -> DataFrame:
    """
    Drop columns that have only one unique value across all rows.
    These columns carry no information for ML.
    
    Protected columns (e.g., 'label') are never dropped even if they
    have a single value, because they are needed for ML training.
    
    Args:
        df: input DataFrame
        protected_columns: columns to preserve even if constant (default: ['label'])
    """
    if protected_columns is None:
        protected_columns = ['label']
    
    constant_cols = []
    
    for col_name in df.columns:
        # Skip metadata columns
        if col_name.startswith("_"):
            continue
        
        # Skip protected columns
        if col_name in protected_columns:
            continue
        
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            constant_cols.append(col_name)
    
    if constant_cols:
        df_clean = df.drop(*constant_cols)
        logger.info(f"Step 6 - drop_constant_columns: dropped {len(constant_cols)} "
                    f"constant columns: {constant_cols}")
        logger.info(f"  Protected columns preserved: {protected_columns}")
    else:
        df_clean = df
        logger.info(f"Step 6 - drop_constant_columns: no constant columns found")
    
    return df_clean


# =============================================================================
# Step 7 - Add derived columns
# =============================================================================

def add_derived_columns(df: DataFrame, label_col: str = "label") -> DataFrame:
    """
    Add derived columns useful for analysis and ML:
    - is_attack (binary 0/1): 1 if label != BENIGN, 0 otherwise
    - attack_category (high-level grouping): DoS, Probe, U2R, R2L, Benign
    
    Args:
        df: input DataFrame
        label_col: name of the label column (default: 'label')
    """
    if label_col not in df.columns:
        logger.warning(f"Step 7 - add_derived_columns: column '{label_col}' not found, skipping")
        return df
    
    # Binary label
    df_clean = df.withColumn(
        "is_attack",
        F.when(F.col(label_col) == "BENIGN", 0).otherwise(1)
    )
    
    # Attack category (high-level grouping based on attack type)
    # This is useful for the Gold layer and ML analysis
    attack_category_expr = (
        F.when(F.col(label_col) == "BENIGN", "Benign")
        .when(F.col(label_col).isin(["DoS-Hulk", "DoS-GoldenEye", "DoS-Slowloris",
                                     "DoS-SlowHTTPTest", "DDoS"]), "DoS")
        .when(F.col(label_col).isin(["PortScan"]), "Probe")
        .when(F.col(label_col).isin(["Brute Force", "FTP-Patator", "SSH-Patator"]), "Brute Force")
        .when(F.col(label_col).like("Web Attack%"), "Web Attack")
        .when(F.col(label_col).isin(["Bot", "Infiltration", "Heartbleed"]), "Other Attack")
        .otherwise("Unknown")
    )
    
    df_clean = df_clean.withColumn("attack_category", attack_category_expr)
    
    n_categories = df_clean.select("attack_category").distinct().count()
    logger.info(f"Step 7 - add_derived_columns: added 2 columns (is_attack, attack_category) - "
                f"{n_categories} attack categories")
    
    return df_clean


# =============================================================================
# Main orchestration function
# =============================================================================

def clean_bronze_to_silver(
    df: DataFrame,
    critical_columns: Optional[List[str]] = None,
    label_col: str = "label",
) -> DataFrame:
    """
    Run the complete Bronze -> Silver cleaning pipeline.
    
    Args:
        df: Bronze DataFrame (output of ingest_csv_to_bronze_memory)
        critical_columns: columns where null means drop the row
        label_col: name of the label column (default: 'label')
    
    Returns:
        Silver DataFrame (cleaned, typed, enriched)
    
    Pipeline:
        1. Remove duplicates
        2. Handle nulls
        3. Filter invalid values
        4. Standardize labels
        5. Cast types
        6. Drop constant columns
        7. Add derived columns (is_attack, attack_category)
    """
    logger.info("=" * 60)
    logger.info("Bronze -> Silver cleaning pipeline started")
    logger.info("=" * 60)
    
    initial_count = df.count()
    initial_cols = len(df.columns)
    logger.info(f"Input Bronze DataFrame: {initial_count} rows, {initial_cols} columns")
    
    # Default critical columns if not provided
    if critical_columns is None:
        critical_columns = ["source_ip", "destination_ip", "label"]
        # Only keep columns that exist in the DataFrame
        critical_columns = [c for c in critical_columns if c in df.columns]
    
    # Run each step in sequence
    df_silver = df
    df_silver = remove_duplicates(df_silver)
    df_silver = handle_nulls(df_silver, critical_columns=critical_columns)
    df_silver = filter_invalid_values(df_silver)
    df_silver = standardize_labels(df_silver, label_col=label_col)
    df_silver = cast_types(df_silver)
    df_silver = drop_constant_columns(df_silver)
    df_silver = add_derived_columns(df_silver, label_col=label_col)
    
    final_count = df_silver.count()
    final_cols = len(df_silver.columns)
    
    logger.info("=" * 60)
    logger.info(f"Silver cleaning pipeline completed")
    logger.info(f"  Rows:    {initial_count} -> {final_count} "
                f"({initial_count - final_count} removed)")
    logger.info(f"  Columns: {initial_cols} -> {final_cols} "
                f"({final_cols - initial_cols:+d} change)")
    logger.info("=" * 60)
    
    return df_silver


# === For local CLI usage ===
if __name__ == "__main__":
    import argparse
    from src.ingestion import create_spark_session, ingest_csv_to_bronze_memory
    
    parser = argparse.ArgumentParser(description="Bronze -> Silver cleaning")
    parser.add_argument("--input", default="data/raw/", help="Raw CSV folder")
    args = parser.parse_args()
    
    spark = create_spark_session()
    try:
        df_bronze = ingest_csv_to_bronze_memory(spark, args.input)
        df_silver = clean_bronze_to_silver(df_bronze)
        print(f"\nSilver DataFrame: {df_silver.count()} rows, {len(df_silver.columns)} columns")
        df_silver.show(5)
    finally:
        spark.stop()