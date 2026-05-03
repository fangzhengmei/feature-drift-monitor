import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from drift_monitor.data_loader import DataLoader


class TestDataLoader:
    def test_load_csv(self, temp_dir):
        loader = DataLoader()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        csv_path = temp_dir / "test.csv"
        df.to_csv(csv_path, index=False)

        result = loader.load(str(csv_path))
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_load_nonexistent_file(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.csv")

    def test_load_unsupported_format(self, temp_dir):
        loader = DataLoader()
        txt_path = temp_dir / "test.txt"
        txt_path.write_text("hello")

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(str(txt_path))

    def test_detect_column_types_numerical(self):
        loader = DataLoader()
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], 100),
            "b": np.random.normal(0, 1, 100),
        })

        types = loader.detect_column_types(df, categorical_threshold=10)
        assert types["a"] == "categorical"
        assert types["b"] == "numerical"

    def test_detect_column_types_categorical(self):
        loader = DataLoader()
        df = pd.DataFrame({
            "a": ["x", "y", "z", "x", "y"],
            "b": pd.Categorical(["A", "B", "A", "B", "A"]),
        })

        types = loader.detect_column_types(df)
        assert types["a"] == "categorical"
        assert types["b"] == "categorical"

    def test_detect_column_types_low_cardinality_numerical(self):
        loader = DataLoader()
        df = pd.DataFrame({
            "a": [1, 2, 3, 1, 2, 3, 1, 2],
        })

        types = loader.detect_column_types(df, categorical_threshold=5)
        assert types["a"] == "categorical"

        types2 = loader.detect_column_types(df, categorical_threshold=2)
        assert types2["a"] == "numerical"

    def test_get_metadata_after_load(self, temp_dir):
        loader = DataLoader()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        csv_path = temp_dir / "test.csv"
        df.to_csv(csv_path, index=False)

        loader.load(str(csv_path))
        metadata = loader.get_metadata()

        assert metadata["rows"] == 3
        assert metadata["columns"] == ["a", "b"]
        assert "dtypes" in metadata
