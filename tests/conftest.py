import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_numerical_data_no_drift():
    np.random.seed(42)
    expected = pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "score": np.random.uniform(0, 100, 1000),
    })
    actual = pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "score": np.random.uniform(0, 100, 1000),
    })
    return expected, actual


@pytest.fixture
def sample_numerical_data_with_drift():
    np.random.seed(42)
    expected = pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
    })
    actual = pd.DataFrame({
        "age": np.random.normal(40, 5, 1000),
        "income": np.random.normal(80000, 15000, 1000),
    })
    return expected, actual


@pytest.fixture
def sample_categorical_data_no_drift():
    np.random.seed(42)
    expected = pd.DataFrame({
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.45, 0.45, 0.1]),
        "region": np.random.choice(["North", "South", "East", "West"], 1000, p=[0.3, 0.3, 0.2, 0.2]),
    })
    actual = pd.DataFrame({
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.45, 0.45, 0.1]),
        "region": np.random.choice(["North", "South", "East", "West"], 1000, p=[0.3, 0.3, 0.2, 0.2]),
    })
    return expected, actual


@pytest.fixture
def sample_categorical_data_with_drift():
    np.random.seed(42)
    expected = pd.DataFrame({
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.45, 0.45, 0.1]),
        "region": np.random.choice(["North", "South", "East", "West"], 1000, p=[0.3, 0.3, 0.2, 0.2]),
    })
    actual = pd.DataFrame({
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.1, 0.8, 0.1]),
        "region": np.random.choice(["North", "South", "East", "West"], 1000, p=[0.1, 0.5, 0.2, 0.2]),
    })
    return expected, actual


@pytest.fixture
def sample_mixed_data():
    np.random.seed(42)
    expected = pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.45, 0.45, 0.1]),
        "region": np.random.choice(["North", "South", "East", "West"], 1000, p=[0.3, 0.3, 0.2, 0.2]),
    })
    actual = pd.DataFrame({
        "age": np.random.normal(40, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.1, 0.8, 0.1]),
        "region": np.random.choice(["North", "South", "East", "West"], 1000, p=[0.3, 0.3, 0.2, 0.2]),
    })
    return expected, actual


@pytest.fixture
def data_with_missing_values():
    np.random.seed(42)
    expected = pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.45, 0.45, 0.1]),
    })
    actual = pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "gender": np.random.choice(["M", "F", "O"], 1000, p=[0.45, 0.45, 0.1]),
    })
    actual.loc[np.random.choice(1000, 100, replace=False), "age"] = np.nan
    actual.loc[np.random.choice(1000, 50, replace=False), "income"] = np.nan
    return expected, actual


@pytest.fixture
def csv_files(temp_dir, sample_mixed_data):
    expected, actual = sample_mixed_data
    expected_path = temp_dir / "expected.csv"
    actual_path = temp_dir / "actual.csv"
    expected.to_csv(expected_path, index=False)
    actual.to_csv(actual_path, index=False)
    return expected_path, actual_path
