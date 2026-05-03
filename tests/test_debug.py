import pytest
import pandas as pd
import numpy as np
from drift_monitor.data_loader import DataLoader


def test_detect_str_dtype():
    df1 = pd.DataFrame({
        'a': ['x', 'y', 'z', 'x', 'y'],
        'b': pd.Categorical(['A', 'B', 'A', 'B', 'A']),
    })

    loader = DataLoader()
    types1 = loader.detect_column_types(df1)

    assert types1['a'] == 'categorical'
    assert types1['b'] == 'categorical'


def test_detect_np_choice_strings():
    np.random.seed(42)
    expected = pd.DataFrame({
        'gender': np.random.choice(['M', 'F', 'O'], 100, p=[0.45, 0.45, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100, p=[0.3, 0.3, 0.2, 0.2]),
    })

    loader = DataLoader()
    types = loader.detect_column_types(expected)

    assert types['gender'] == 'categorical'
    assert types['region'] == 'categorical'
