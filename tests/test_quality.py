"""
Tests for the cleaning and transformation modules.

Covers:
- src/cleaning.py functions (remove_duplicates, handle_nulls, etc.)
- src/transformation.py functions (add_packet_aggregate_features, etc.)
- End-to-end pipeline (Bronze -> Silver -> Gold)
"""

import pytest

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, FloatType
)

from src.cleaning import (
    remove_duplicates,
    handle_nulls,
    filter_invalid_values,
    standardize_labels,
    cast_types,
    drop_constant_columns,
    add_derived_columns,
    clean_bronze_to_silver,
)
from src.transformation import (
    add_packet_aggregate_features,
    add_flag_aggregate_features,
    add_iat_features,
    add_protocol_name,
    add_binary_target,
    add_flow_efficiency_features,
    transform_silver_to_gold,
)


# =============================================================================
# Tests for cleaning functions
# =============================================================================

class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self, spark):
        schema = StructType([
            StructField("a", StringType(), True),
            StructField("b", IntegerType(), True),
        ])
        data = [("x", 1), ("y", 2), ("x", 1), ("z", 3), ("y", 2)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = remove_duplicates(df)
        
        assert df_clean.count() == 3

    def test_no_duplicates_returns_same_count(self, spark):
        schema = StructType([StructField("a", StringType(), True)])
        df = spark.createDataFrame([("x",), ("y",), ("z",)], schema)
        
        df_clean = remove_duplicates(df)
        
        assert df_clean.count() == 3


class TestHandleNulls:
    def test_drops_rows_with_critical_nulls(self, spark):
        schema = StructType([
            StructField("source_ip", StringType(), True),
            StructField("label", StringType(), True),
        ])
        data = [("192.168.1.1", "BENIGN"), (None, "DDoS"), ("10.0.0.5", None)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = handle_nulls(df, critical_columns=["source_ip", "label"])
        
        assert df_clean.count() == 1

    def test_fills_non_critical_nulls(self, spark):
        schema = StructType([
            StructField("source_ip", StringType(), True),
            StructField("protocol", StringType(), True),
        ])
        data = [("192.168.1.1", "TCP"), ("10.0.0.5", None)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = handle_nulls(df, critical_columns=["source_ip"], fill_value="unknown")
        
        assert df_clean.count() == 2
        # The null in 'protocol' should be filled with 'unknown'
        assert df_clean.filter(F.col("protocol") == "unknown").count() == 1


class TestFilterInvalidValues:
    def test_filters_negative_bytes(self, spark):
        schema = StructType([
            StructField("source_ip", StringType(), True),
            StructField("bytes", LongType(), True),
        ])
        data = [("192.168.1.1", 1000), ("10.0.0.5", -500), ("10.0.0.6", 200)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = filter_invalid_values(df)
        
        assert df_clean.count() == 2

    def test_filters_invalid_ports(self, spark):
        schema = StructType([
            StructField("source_ip", StringType(), True),
            StructField("port", IntegerType(), True),
        ])
        data = [("192.168.1.1", 80), ("10.0.0.5", 70000), ("10.0.0.6", 443)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = filter_invalid_values(df)
        
        assert df_clean.count() == 2


class TestStandardizeLabels:
    def test_trims_whitespace(self, spark):
        schema = StructType([StructField("label", StringType(), True)])
        data = [("  BENIGN  ",), ("DDoS ",), (" DoS-Hulk",)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = standardize_labels(df)
        
        labels = [row.label for row in df_clean.collect()]
        assert "BENIGN" in labels
        assert "DDoS" in labels
        assert "DoS-Hulk" in labels
        # No label should have leading/trailing whitespace
        for label in labels:
            assert label == label.strip()

    def test_handles_missing_label_column(self, spark):
        schema = StructType([StructField("foo", StringType(), True)])
        df = spark.createDataFrame([("bar",)], schema)
        
        df_clean = standardize_labels(df, label_col="label")
        
        # Should return the DataFrame unchanged (just log a warning)
        assert df_clean.count() == 1


class TestDropConstantColumns:
    def test_preserves_label_column(self, spark):
        """The label column must NEVER be dropped, even if constant."""
        schema = StructType([
            StructField("source_ip", StringType(), True),
            StructField("label", StringType(), True),  # constant
            StructField("always_zero", IntegerType(), True),  # constant
        ])
        data = [
            ("192.168.1.1", "BENIGN", 0),
            ("10.0.0.5", "BENIGN", 0),
            ("172.16.0.1", "BENIGN", 0),
        ]
        df = spark.createDataFrame(data, schema)
        
        df_clean = drop_constant_columns(df)
        
        assert "label" in df_clean.columns  # CRITICAL: label preserved
        assert "always_zero" not in df_clean.columns  # constant dropped
        assert "source_ip" in df_clean.columns  # non-constant preserved

    def test_drops_only_truly_constant_columns(self, spark):
        schema = StructType([
            StructField("a", StringType(), True),
            StructField("b", IntegerType(), True),
        ])
        data = [("x", 1), ("y", 2), ("z", 3)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = drop_constant_columns(df)
        
        # No constant columns, nothing dropped
        assert df_clean.count() == 3
        assert len(df_clean.columns) == 2


class TestAddDerivedColumns:
    def test_adds_is_attack_column(self, spark):
        schema = StructType([StructField("label", StringType(), True)])
        data = [("BENIGN",), ("DDoS",), ("PortScan",), ("BENIGN",)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = add_derived_columns(df)
        
        assert "is_attack" in df_clean.columns
        # 2 BENIGN -> is_attack=0, 2 attacks -> is_attack=1
        assert df_clean.filter(F.col("is_attack") == 0).count() == 2
        assert df_clean.filter(F.col("is_attack") == 1).count() == 2

    def test_adds_attack_category_column(self, spark):
        schema = StructType([StructField("label", StringType(), True)])
        data = [("BENIGN",), ("DDoS",), ("DoS-Hulk",), ("PortScan",)]
        df = spark.createDataFrame(data, schema)
        
        df_clean = add_derived_columns(df)
        
        assert "attack_category" in df_clean.columns
        categories = [row.attack_category for row in df_clean.collect()]
        assert "Benign" in categories
        assert "DoS" in categories
        assert "Probe" in categories


# =============================================================================
# Tests for transformation functions
# =============================================================================

class TestAddPacketAggregateFeatures:
    def test_adds_5_new_features(self, spark):
        schema = StructType([
            StructField("total_fwd_packets", IntegerType(), True),
            StructField("total_backward_packets", IntegerType(), True),
            StructField("fwd_packets_length_total", FloatType(), True),
            StructField("bwd_packets_length_total", FloatType(), True),
        ])
        data = [(10, 5, 1000.0, 500.0)]
        df = spark.createDataFrame(data, schema)
        
        df_gold = add_packet_aggregate_features(df)
        
        assert "total_packets" in df_gold.columns
        assert "total_bytes" in df_gold.columns
        assert "fwd_bwd_packet_ratio" in df_gold.columns
        assert "fwd_bwd_bytes_ratio" in df_gold.columns
        assert "bytes_per_packet" in df_gold.columns

    def test_total_packets_is_sum(self, spark):
        schema = StructType([
            StructField("total_fwd_packets", IntegerType(), True),
            StructField("total_backward_packets", IntegerType(), True),
        ])
        data = [(10, 5)]
        df = spark.createDataFrame(data, schema)
        
        df_gold = add_packet_aggregate_features(df)
        
        result = df_gold.select("total_packets").first()
        assert result.total_packets == 15

    def test_handles_division_by_zero(self, spark):
        """fwd_bwd_packet_ratio should be 0 when total_backward_packets is 0."""
        schema = StructType([
            StructField("total_fwd_packets", IntegerType(), True),
            StructField("total_backward_packets", IntegerType(), True),
        ])
        data = [(10, 0)]
        df = spark.createDataFrame(data, schema)
        
        df_gold = add_packet_aggregate_features(df)
        
        result = df_gold.select("fwd_bwd_packet_ratio").first()
        assert result.fwd_bwd_packet_ratio == 0.0


class TestAddBinaryTarget:
    def test_benign_becomes_zero(self, spark):
        schema = StructType([StructField("label", StringType(), True)])
        data = [("BENIGN",), ("Benign",), ("BENIGN",)]
        df = spark.createDataFrame(data, schema)
        
        df_gold = add_binary_target(df)
        
        # All BENIGN -> all is_attack = 0
        assert df_gold.filter(F.col("is_attack") == 0).count() == 3

    def test_attacks_become_one(self, spark):
        schema = StructType([StructField("label", StringType(), True)])
        data = [("DDoS",), ("DoS-Hulk",), ("PortScan",)]
        df = spark.createDataFrame(data, schema)
        
        df_gold = add_binary_target(df)
        
        # All attacks -> all is_attack = 1
        assert df_gold.filter(F.col("is_attack") == 1).count() == 3

    def test_handles_missing_label(self, spark):
        schema = StructType([StructField("foo", StringType(), True)])
        df = spark.createDataFrame([("bar",)], schema)
        
        df_gold = add_binary_target(df, label_col="label")
        
        # Should return unchanged (just log a warning)
        assert "is_attack" not in df_gold.columns


class TestAddProtocolName:
    def test_maps_tcp(self, spark):
        schema = StructType([StructField("protocol", IntegerType(), True)])
        df = spark.createDataFrame([(6,)], schema)
        
        df_gold = add_protocol_name(df)
        
        result = df_gold.select("protocol_name").first()
        assert result.protocol_name == "TCP"

    def test_maps_udp(self, spark):
        schema = StructType([StructField("protocol", IntegerType(), True)])
        df = spark.createDataFrame([(17,)], schema)
        
        df_gold = add_protocol_name(df)
        
        result = df_gold.select("protocol_name").first()
        assert result.protocol_name == "UDP"

    def test_maps_icmp(self, spark):
        schema = StructType([StructField("protocol", IntegerType(), True)])
        df = spark.createDataFrame([(1,)], schema)
        
        df_gold = add_protocol_name(df)
        
        result = df_gold.select("protocol_name").first()
        assert result.protocol_name == "ICMP"


# =============================================================================
# End-to-end pipeline tests
# =============================================================================

class TestEndToEndPipeline:
    """Tests for the complete Bronze -> Silver -> Gold pipeline."""

    def test_bronze_to_silver_preserves_label(self, bronze_df):
        """The Silver layer must still contain the label column."""
        df_silver = clean_bronze_to_silver(bronze_df)
        
        assert "label" in df_silver.columns
        # Should have removed the duplicate
        assert df_silver.count() == 4  # 5 rows - 1 duplicate

    def test_silver_to_gold_adds_features(self, bronze_df):
        """The Gold layer should have AT LEAST 10 new features compared to Silver."""
        df_silver = clean_bronze_to_silver(bronze_df)
        df_gold = transform_silver_to_gold(df_silver)
        
        new_features = [c for c in df_gold.columns if c not in df_silver.columns]
        # Expected: 5 packet + 3 flag + 4 IAT + 1 protocol_name + 1 is_attack + 4 flow_efficiency = 18 max
        # We accept 10+ since some may be skipped if columns are dropped
        assert len(new_features) >= 10, f"Expected >= 10 new features, got {len(new_features)}: {new_features}"

    def test_gold_has_is_attack(self, bronze_df):
        """The Gold layer must have the is_attack binary target with both 0 and 1 values."""
        df_silver = clean_bronze_to_silver(bronze_df)
        df_gold = transform_silver_to_gold(df_silver)
        
        assert "is_attack" in df_gold.columns
        # Should have both 0 (benign) and 1 (attack) values
        benign_count = df_gold.filter(F.col("is_attack") == 0).count()
        attack_count = df_gold.filter(F.col("is_attack") == 1).count()
        assert benign_count > 0, f"Expected at least 1 benign row, got {benign_count}"
        assert attack_count > 0, f"Expected at least 1 attack row, got {attack_count}"

    def test_gold_has_protocol_name(self, bronze_df):
        """The Gold layer should have the protocol_name categorical column."""
        df_silver = clean_bronze_to_silver(bronze_df)
        df_gold = transform_silver_to_gold(df_silver)
        
        assert "protocol_name" in df_gold.columns