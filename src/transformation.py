"""
Silver -> Gold transformation pipeline for the CIC-IDS2017 (DistriNet) dataset.

Applies feature engineering to produce a Gold layer ready for
Machine Learning training and Business Intelligence analytics.

Each transformation is a standalone function so it can be tested individually.
The main entry point is `transform_silver_to_gold()` which orchestrates all steps.

Databricks usage:
    from src.transformation import transform_silver_to_gold
    df_gold = transform_silver_to_gold(df_silver)
"""

import logging
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit

# === Logging ===
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1 - Packet aggregate features
# =============================================================================

def add_packet_aggregate_features(df: DataFrame) -> DataFrame:
    """
    Add aggregate features at the packet level:
    - total_packets: total fwd + bwd packets
    - total_bytes: total fwd + bwd bytes
    - fwd_bwd_packet_ratio: ratio of forward to backward packets
    - fwd_bwd_bytes_ratio: ratio of forward to backward bytes
    - bytes_per_packet: average bytes per packet
    """
    df_gold = df
    
    # Total packets
    if "total_fwd_packets" in df.columns and "total_backward_packets" in df.columns:
        df_gold = df_gold.withColumn(
            "total_packets",
            F.col("total_fwd_packets") + F.col("total_backward_packets")
        )
    
    # Total bytes
    if "fwd_packets_length_total" in df.columns and "bwd_packets_length_total" in df.columns:
        df_gold = df_gold.withColumn(
            "total_bytes",
            F.col("fwd_packets_length_total") + F.col("bwd_packets_length_total")
        )
    
    # Fwd/Bwd packet ratio (with division-by-zero protection)
    if "total_fwd_packets" in df.columns and "total_backward_packets" in df.columns:
        df_gold = df_gold.withColumn(
            "fwd_bwd_packet_ratio",
            F.when(F.col("total_backward_packets") == 0, F.lit(0.0))
             .otherwise(F.col("total_fwd_packets") / F.col("total_backward_packets"))
        )
    
    # Fwd/Bwd bytes ratio
    if "fwd_packets_length_total" in df.columns and "bwd_packets_length_total" in df.columns:
        df_gold = df_gold.withColumn(
            "fwd_bwd_bytes_ratio",
            F.when(F.col("bwd_packets_length_total") == 0, F.lit(0.0))
             .otherwise(F.col("fwd_packets_length_total") / F.col("bwd_packets_length_total"))
        )
    
    # Bytes per packet
    if "total_packets" in df_gold.columns and "total_bytes" in df_gold.columns:
        df_gold = df_gold.withColumn(
            "bytes_per_packet",
            F.when(F.col("total_packets") == 0, F.lit(0.0))
             .otherwise(F.col("total_bytes") / F.col("total_packets"))
        )
    
    logger.info("Step 1 - add_packet_aggregate_features: added 5 features "
                "(total_packets, total_bytes, fwd_bwd_packet_ratio, "
                "fwd_bwd_bytes_ratio, bytes_per_packet)")
    
    return df_gold


# =============================================================================
# Step 2 - Flag aggregate features
# =============================================================================

def add_flag_aggregate_features(df: DataFrame) -> DataFrame:
    """
    Add aggregate features at the TCP flag level:
    - total_flag_count: sum of all TCP flag counts
    - has_syn_flag: binary indicator for SYN flag (often used in DoS)
    - has_rst_flag: binary indicator for RST flag (often used in scans)
    """
    flag_cols = [
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
        'ack_flag_count', 'cwr_flag_count', 'ece_flag_count'
    ]
    
    available_flags = [c for c in flag_cols if c in df.columns]
    
    df_gold = df
    
    if available_flags:
        # Sum of all flags
        df_gold = df_gold.withColumn(
            "total_flag_count",
            sum(F.col(c) for c in available_flags)
        )
        
        # Binary indicators (1 if flag count > 0, 0 otherwise)
        if "syn_flag_count" in available_flags:
            df_gold = df_gold.withColumn(
                "has_syn_flag",
                F.when(F.col("syn_flag_count") > 0, 1).otherwise(0)
            )
        
        if "rst_flag_count" in available_flags:
            df_gold = df_gold.withColumn(
                "has_rst_flag",
                F.when(F.col("rst_flag_count") > 0, 1).otherwise(0)
            )
        
        logger.info(f"Step 2 - add_flag_aggregate_features: added 3 features "
                    f"(total_flag_count, has_syn_flag, has_rst_flag) from {len(available_flags)} flag columns")
    
    return df_gold


# =============================================================================
# Step 3 - Inter-Arrival Time (IAT) features
# =============================================================================

def add_iat_features(df: DataFrame) -> DataFrame:
    """
    Add Inter-Arrival Time (IAT) derived features:
    - flow_iat_range: max - min IAT for the flow
    - active_range: max - min active time
    - idle_range: max - min idle time
    - idle_active_ratio: ratio of idle to active time
    """
    df_gold = df
    
    # Flow IAT range
    if "flow_iat_max" in df.columns and "flow_iat_min" in df.columns:
        df_gold = df_gold.withColumn(
            "flow_iat_range",
            F.col("flow_iat_max") - F.col("flow_iat_min")
        )
    
    # Active range
    if "active_max" in df.columns and "active_min" in df.columns:
        df_gold = df_gold.withColumn(
            "active_range",
            F.col("active_max") - F.col("active_min")
        )
    
    # Idle range
    if "idle_max" in df.columns and "idle_min" in df.columns:
        df_gold = df_gold.withColumn(
            "idle_range",
            F.col("idle_max") - F.col("idle_min")
        )
    
    # Idle/Active ratio (with division-by-zero protection)
    if "idle_mean" in df.columns and "active_mean" in df.columns:
        df_gold = df_gold.withColumn(
            "idle_active_ratio",
            F.when(F.col("active_mean") == 0, F.lit(0.0))
             .otherwise(F.col("idle_mean") / F.col("active_mean"))
        )
    
    logger.info("Step 3 - add_iat_features: added 4 features "
                "(flow_iat_range, active_range, idle_range, idle_active_ratio)")
    
    return df_gold


# =============================================================================
# Step 4 - Protocol name (categorical)
# =============================================================================

def add_protocol_name(df: DataFrame) -> DataFrame:
    """
    Add a human-readable protocol name based on the protocol number.
    
    Mapping:
    - 6  -> TCP
    - 17 -> UDP
    - 1  -> ICMP
    - 0  -> Other
    """
    if "protocol" not in df.columns:
        logger.warning("Step 4 - add_protocol_name: 'protocol' column not found, skipping")
        return df
    
    df_gold = df.withColumn(
        "protocol_name",
        F.when(F.col("protocol") == 6, F.lit("TCP"))
         .when(F.col("protocol") == 17, F.lit("UDP"))
         .when(F.col("protocol") == 1, F.lit("ICMP"))
         .when(F.col("protocol") == 0, F.lit("Other"))
         .otherwise(F.lit("Other"))
    )
    
    logger.info("Step 4 - add_protocol_name: added protocol_name (categorical)")
    
    return df_gold


# =============================================================================
# Step 5 - Binary target for ML
# =============================================================================

def add_binary_target(df: DataFrame, label_col: str = "label") -> DataFrame:
    """
    Add a binary target column `is_attack` based on the label:
    - 0 if label is "Benign" (case-insensitive: BENIGN, Benign, benign)
    - 1 otherwise (any attack type)
    
    This is the target variable for binary classification in ML.
    
    Case-insensitive matching is used because CIC-IDS2017 variations
    exist across dataset versions ("BENIGN", "Benign", "benign").
    """
    if label_col not in df.columns:
        logger.warning(f"Step 5 - add_binary_target: column '{label_col}' not found, skipping")
        return df
    
    # Case-insensitive comparison: lowercase the label and compare to "benign"
    df_gold = df.withColumn(
        "is_attack",
        F.when(F.lower(F.col(label_col)) == "benign", 0).otherwise(1)
    )
    
    # Stats
    if "is_attack" in df_gold.columns:
        benign_count = df_gold.filter(F.col("is_attack") == 0).count()
        attack_count = df_gold.filter(F.col("is_attack") == 1).count()
        total = benign_count + attack_count
        if total > 0:
            attack_ratio = attack_count / total * 100
            logger.info(f"Step 5 - add_binary_target: added is_attack "
                        f"(Benign: {benign_count}, Attack: {attack_count}, "
                        f"Attack ratio: {attack_ratio:.1f}%)")
    
    return df_gold


# =============================================================================
# Step 6 - Flow efficiency metrics
# =============================================================================

def add_flow_efficiency_features(df: DataFrame) -> DataFrame:
    """
    Add flow efficiency metrics:
    - packet_rate: packets per second (total_packets / flow_duration)
    - byte_rate: bytes per second (total_bytes / flow_duration)
    - header_payload_ratio: ratio of header to payload bytes
    """
    df_gold = df
    
    # Packet rate (packets per microsecond)
    if "total_packets" in df.columns and "flow_duration" in df.columns:
        df_gold = df_gold.withColumn(
            "packet_rate",
            F.when(F.col("flow_duration") == 0, F.lit(0.0))
             .otherwise(F.col("total_packets") / F.col("flow_duration"))
        )
    
    # Byte rate
    if "total_bytes" in df.columns and "flow_duration" in df.columns:
        df_gold = df_gold.withColumn(
            "byte_rate",
            F.when(F.col("flow_duration") == 0, F.lit(0.0))
             .otherwise(F.col("total_bytes") / F.col("flow_duration"))
        )
    
    # Header/Payload ratio
    if "fwd_header_length" in df.columns and "bwd_header_length" in df.columns:
        df_gold = df_gold.withColumn(
            "total_header_length",
            F.col("fwd_header_length") + F.col("bwd_header_length")
        )
        
        if "total_bytes" in df_gold.columns:
            df_gold = df_gold.withColumn(
                "header_payload_ratio",
                F.when(F.col("total_bytes") == 0, F.lit(0.0))
                 .otherwise(F.col("total_header_length") / F.col("total_bytes"))
            )
    
    logger.info("Step 6 - add_flow_efficiency_features: added 3-4 features "
                "(packet_rate, byte_rate, total_header_length, header_payload_ratio)")
    
    return df_gold


# =============================================================================
# Main orchestration function
# =============================================================================

def transform_silver_to_gold(
    df: DataFrame,
    label_col: str = "label",
) -> DataFrame:
    """
    Run the complete Silver -> Gold feature engineering pipeline.
    
    Args:
        df: Silver DataFrame (output of clean_bronze_to_silver)
        label_col: name of the label column (default: 'label')
    
    Returns:
        Gold DataFrame with engineered features, ready for ML and BI
    
    Pipeline:
        1. Packet aggregate features (totals, ratios, bytes_per_packet)
        2. Flag aggregate features (total_flag_count, has_syn, has_rst)
        3. IAT features (ranges, idle_active_ratio)
        4. Protocol name (categorical encoding)
        5. Binary target (is_attack: 0/1)
        6. Flow efficiency metrics (packet_rate, byte_rate, header_payload_ratio)
    """
    logger.info("=" * 60)
    logger.info("Silver -> Gold transformation pipeline started")
    logger.info("=" * 60)
    
    initial_count = df.count()
    initial_cols = len(df.columns)
    logger.info(f"Input Silver DataFrame: {initial_count} rows, {initial_cols} columns")
    
    df_gold = df
    df_gold = add_packet_aggregate_features(df_gold)
    df_gold = add_flag_aggregate_features(df_gold)
    df_gold = add_iat_features(df_gold)
    df_gold = add_protocol_name(df_gold)
    df_gold = add_binary_target(df_gold, label_col=label_col)
    df_gold = add_flow_efficiency_features(df_gold)
    
    final_count = df_gold.count()
    final_cols = len(df_gold.columns)
    
    logger.info("=" * 60)
    logger.info("Gold transformation pipeline completed")
    logger.info(f"  Rows:    {initial_count} -> {final_count} (no rows lost in feature engineering)")
    logger.info(f"  Columns: {initial_cols} -> {final_cols} (+{final_cols - initial_cols} new features)")
    logger.info("=" * 60)
    
    return df_gold


# === For local CLI usage ===
if __name__ == "__main__":
    import argparse
    from src.ingestion import create_spark_session, ingest_parquet_to_bronze_memory
    from src.cleaning import clean_bronze_to_silver
    
    parser = argparse.ArgumentParser(description="Silver -> Gold transformation")
    parser.add_argument("--input", required=True, help="Parquet file path")
    args = parser.parse_args()
    
    spark = create_spark_session()
    try:
        df_bronze = ingest_parquet_to_bronze_memory(spark, args.input)
        df_silver = clean_bronze_to_silver(df_bronze)
        df_gold = transform_silver_to_gold(df_silver)
        print(f"\nGold DataFrame: {df_gold.count()} rows, {len(df_gold.columns)} columns")
        df_gold.show(5)
    finally:
        spark.stop()