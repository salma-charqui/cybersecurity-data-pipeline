"""
Tests for the ingestion module (src/ingestion.py).

Covers:
- clean_column_name (pure function, no Spark needed)
- ingest_csv_to_bronze_memory (Spark required)
"""

import os
import tempfile

import pytest

from src.ingestion import (
    clean_column_name,
    ingest_csv_to_bronze_memory,
    generate_sample_dataset,
)


# =============================================================================
# Tests for clean_column_name (pure function)
# =============================================================================

class TestCleanColumnName:
    """Tests for the clean_column_name function."""

    def test_strips_leading_whitespace(self):
        assert clean_column_name(" Source IP") == "source_ip"

    def test_strips_trailing_whitespace(self):
        assert clean_column_name("Destination Port ") == "destination_port"

    def test_converts_internal_spaces_to_underscores(self):
        assert clean_column_name("Flow Duration") == "flow_duration"
        assert clean_column_name("Total Fwd Packets") == "total_fwd_packets"

    def test_converts_to_lowercase(self):
        assert clean_column_name("SourceIP") == "sourceip"
        assert clean_column_name("PROTOCOL") == "protocol"

    def test_handles_slashes(self):
        assert clean_column_name("Source/IP") == "source_ip"

    def test_handles_dashes(self):
        assert clean_column_name("DoS-Hulk") == "dos_hulk"

    def test_handles_already_clean_names(self):
        assert clean_column_name("source_ip") == "source_ip"

    def test_handles_empty_string(self):
        assert clean_column_name("") == ""

    def test_handles_real_cic_ids2017_columns(self):
        """Test with real CIC-IDS2017 column names that have known issues."""
        assert clean_column_name(" Source IP") == "source_ip"
        assert clean_column_name(" Destination Port") == "destination_port"
        assert clean_column_name("Flow Bytes/s") == "flow_bytes_s"
        assert clean_column_name("Fwd Packet Length Max") == "fwd_packet_length_max"


# =============================================================================
# Tests for ingest_csv_to_bronze_memory (requires Spark)
# =============================================================================

class TestIngestCsvToBronzeMemory:
    """Tests for the ingest_csv_to_bronze_memory function."""

    def test_returns_dataframe(self, spark, tmp_path):
        """Should return a non-null DataFrame."""
        # Create a small CSV file
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("source_ip,destination_ip,label\n192.168.1.1,8.8.8.8,BENIGN\n10.0.0.5,8.8.8.8,DDoS\n")
        
        df = ingest_csv_to_bronze_memory(spark, str(tmp_path))
        
        assert df is not None
        assert df.count() == 2

    def test_adds_metadata_columns(self, spark, tmp_path):
        """Should add _source_file and _ingestion_ts metadata columns."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("source_ip,label\n192.168.1.1,BENIGN\n")
        
        df = ingest_csv_to_bronze_memory(spark, str(tmp_path))
        
        assert "_source_file" in df.columns
        assert "_ingestion_ts" in df.columns

    def test_cleans_column_names(self, spark, tmp_path):
        """Should clean column names (e.g., 'Source IP' -> 'source_ip')."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Source IP,Destination Port,Label\n192.168.1.1,80,BENIGN\n")
        
        df = ingest_csv_to_bronze_memory(spark, str(tmp_path))
        
        assert "source_ip" in df.columns
        assert "destination_port" in df.columns
        assert "label" in df.columns
        assert "Source IP" not in df.columns


# =============================================================================
# Tests for generate_sample_dataset
# =============================================================================

class TestGenerateSampleDataset:
    """Tests for the generate_sample_dataset function."""

    def test_creates_csv_file(self, tmp_path):
        """Should create a CSV file in the specified path."""
        output_path = str(tmp_path) + "/"
        generate_sample_dataset(output_path, n_rows=100)
        
        csv_file = os.path.join(str(tmp_path), "cic_sample.csv")
        assert os.path.exists(csv_file)

    def test_csv_has_correct_row_count(self, tmp_path):
        """The CSV should have the requested number of rows."""
        import pandas as pd
        
        output_path = str(tmp_path) + "/"
        generate_sample_dataset(output_path, n_rows=100)
        
        csv_file = os.path.join(str(tmp_path), "cic_sample.csv")
        df = pd.read_csv(csv_file)
        
        assert len(df) == 100

    def test_csv_has_expected_columns(self, tmp_path):
        """The CSV should have the expected columns."""
        import pandas as pd
        
        output_path = str(tmp_path) + "/"
        generate_sample_dataset(output_path, n_rows=10)
        
        csv_file = os.path.join(str(tmp_path), "cic_sample.csv")
        df = pd.read_csv(csv_file)
        
        expected_cols = {"source_ip", "destination_ip", "protocol", "port", "bytes", "label", "timestamp"}
        assert expected_cols == set(df.columns)