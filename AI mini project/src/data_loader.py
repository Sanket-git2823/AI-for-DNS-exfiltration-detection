"""
Robust dataset loader for DNS exfiltration datasets.
Supports CSV loading with encoding fallback, row limiting, and validation.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

class DataLoader:
    """Safely loads and inspects external or synthetic tabular datasets."""
    
    SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "utf-16"]

    @classmethod
    def load_csv(
        cls, 
        file_path: str | Path, 
        max_rows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Loads a CSV file attempting multiple encodings.
        Returns: (DataFrame, encoding_used)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
            
        last_err = None
        for enc in cls.SUPPORTED_ENCODINGS:
            try:
                df = pd.read_csv(path, nrows=max_rows, encoding=enc, low_memory=False)
                if df.empty:
                    raise ValueError(f"File at {path.name} is empty.")
                return df, enc
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_err = e
                continue
                
        raise ValueError(f"Failed to decode {path.name} with supported encodings. Last error: {last_err}")

    @classmethod
    def preview_columns(cls, file_path: str | Path) -> list[str]:
        """Reads only the header line to inspect column names fast."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path, nrows=0)
        return list(df.columns)
