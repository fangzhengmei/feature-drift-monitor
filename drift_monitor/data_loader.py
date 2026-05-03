import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader:
    SUPPORTED_FORMATS = {".csv", ".parquet", ".xlsx", ".xls"}

    def __init__(self):
        self.metadata: Dict[str, Any] = {}

    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")

        if suffix == ".csv":
            df = pd.read_csv(path, **kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(path, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self.metadata = {
            "file_path": str(path),
            "format": suffix,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
        }

        return df

    def detect_column_types(self, df: pd.DataFrame, categorical_threshold: int = 10) -> Dict[str, str]:
        column_types = {}
        for col in df.columns:
            col_dtype = df[col].dtype
            col_dtype_name = col_dtype.name

            if col_dtype_name == "category":
                column_types[col] = "categorical"
            elif col_dtype_name == "object":
                column_types[col] = "categorical"
            elif col_dtype_name == "str":
                column_types[col] = "categorical"
            elif col_dtype_name.startswith("string"):
                column_types[col] = "categorical"
            elif col_dtype_name == "bool":
                column_types[col] = "categorical"
            elif col_dtype_name in ["int64", "int32", "float64", "float32", "Int64", "Int32", "Float64", "Float32"]:
                n_unique = df[col].nunique(dropna=True)
                if n_unique <= categorical_threshold and n_unique > 0:
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "numerical"
            elif pd.api.types.is_integer_dtype(col_dtype):
                n_unique = df[col].nunique(dropna=True)
                if n_unique <= categorical_threshold and n_unique > 0:
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "numerical"
            elif pd.api.types.is_float_dtype(col_dtype):
                n_unique = df[col].nunique(dropna=True)
                if n_unique <= categorical_threshold and n_unique > 0:
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "numerical"
            else:
                column_types[col] = "other"
        return column_types

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
