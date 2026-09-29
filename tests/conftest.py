"""
Shared pytest fixtures for the cybersecurity pipeline tests.

These fixtures provide:
- A SparkSession for tests
- A small synthetic Bronze DataFrame for testing
- A small Silver DataFrame (post-cleaning) for testing
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, FloatType, DoubleType, TimestampType
)


@pytest.fixture(scope="session")
def spark():
    """Provide a SparkSession for the entire test session."""
    spark = (
        SparkSession.builder
        .appName("cybersecurity-pipeline-tests")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    yield spark
    spark.stop()


@pytest.fixture
def bronze_df(spark):
    """
    Small synthetic Bronze DataFrame with all columns needed for
    transformation tests (packet, flag, IAT, header columns).
    Used for testing cleaning and transformation functions.
    """
    schema = StructType([
        StructField("source_ip", StringType(), True),
        StructField("destination_ip", StringType(), True),
        StructField("protocol", IntegerType(), True),  # 6=TCP, 17=UDP, 1=ICMP
        StructField("port", IntegerType(), True),
        StructField("bytes", LongType(), True),
        StructField("label", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        # Packet columns
        StructField("total_fwd_packets", IntegerType(), True),
        StructField("total_backward_packets", IntegerType(), True),
        StructField("fwd_packets_length_total", FloatType(), True),
        StructField("bwd_packets_length_total", FloatType(), True),
        # Flag columns
        StructField("fin_flag_count", IntegerType(), True),
        StructField("syn_flag_count", IntegerType(), True),
        StructField("rst_flag_count", IntegerType(), True),
        StructField("psh_flag_count", IntegerType(), True),
        StructField("ack_flag_count", IntegerType(), True),
        StructField("cwr_flag_count", IntegerType(), True),
        StructField("ece_flag_count", IntegerType(), True),
        # IAT columns
        StructField("flow_iat_max", FloatType(), True),
        StructField("flow_iat_min", FloatType(), True),
        StructField("active_max", FloatType(), True),
        StructField("active_min", FloatType(), True),
        StructField("idle_max", FloatType(), True),
        StructField("idle_min", FloatType(), True),
        StructField("idle_mean", FloatType(), True),
        StructField("active_mean", FloatType(), True),
        # Header columns
        StructField("fwd_header_length", IntegerType(), True),
        StructField("bwd_header_length", IntegerType(), True),
        StructField("flow_duration", LongType(), True),
        # Metadata
        StructField("_source_file", StringType(), True),
        StructField("_ingestion_ts", TimestampType(), True),
    ])
    
    data = [
        # 5 rows, 1 is a duplicate
        ("192.168.1.1", "8.8.8.8", 6, 80, 1000, "Benign", None,
         10, 5, 500.0, 500.0, 0, 0, 0, 0, 5, 0, 0, 100.0, 10.0, 50.0, 10.0, 0.0, 0.0, 25.0, 25.0,
         100, 100, 1000, "test.csv", None),
        ("10.0.0.5", "8.8.8.8", 17, 53, 500, "DDoS", None,
         100, 0, 5000.0, 0.0, 0, 50, 0, 0, 0, 0, 0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         200, 0, 5000, "test.csv", None),
        ("192.168.1.1", "1.1.1.1", 6, 443, 2000, "Benign", None,
         20, 10, 1000.0, 1000.0, 0, 0, 0, 0, 10, 0, 0, 150.0, 20.0, 75.0, 20.0, 0.0, 0.0, 35.0, 35.0,
         150, 150, 2000, "test.csv", None),
        ("172.16.0.1", "8.8.8.8", 1, 0, 100, "PortScan", None,
         5, 5, 50.0, 50.0, 0, 0, 5, 0, 0, 0, 0, 50.0, 10.0, 25.0, 10.0, 0.0, 0.0, 12.5, 12.5,
         50, 50, 100, "test.csv", None),
        ("10.0.0.5", "8.8.8.8", 17, 53, 500, "DDoS", None,
         100, 0, 5000.0, 0.0, 0, 50, 0, 0, 0, 0, 0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         200, 0, 5000, "test.csv", None),  # duplicate
    ]
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def silver_df(spark):
    """
    Small Silver DataFrame (cleaned) for testing transformations.
    Has 4 rows with 2 attacks and 2 benign, no duplicates.
    """
    schema = StructType([
        StructField("source_ip", StringType(), True),
        StructField("destination_ip", StringType(), True),
        StructField("protocol", StringType(), True),
        StructField("port", IntegerType(), True),
        StructField("bytes", LongType(), True),
        StructField("label", StringType(), True),
        StructField("timestamp", TimestampType(), True),
    ])
    
    data = [
        ("192.168.1.1", "8.8.8.8", "TCP", 80, 1000, "BENIGN", None),
        ("10.0.0.5", "8.8.8.8", "UDP", 53, 500, "DDoS", None),
        ("172.16.0.1", "8.8.8.8", "ICMP", 0, 100, "PortScan", None),
        ("192.168.1.4", "1.1.1.1", "TCP", 443, 2000, "BENIGN", None),
    ]
    
    return spark.createDataFrame(data, schema)