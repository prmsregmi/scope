"""Dataset loading and validation."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetLoader:
    """Load and validate conversation datasets."""

    REQUIRED_COLUMNS = ["Chatroom", "Sender", "Timestamp", "Text"]

    def __init__(self, file_path: str) -> None:
        """Initialize dataset loader.

        Args:
            file_path: Path to CSV file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

    def load(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load dataset with optional date filtering.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format (inclusive)
            end_date: End date in 'YYYY-MM-DD' format (inclusive)

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If dataset is invalid
        """
        # Load CSV
        df = pd.read_csv(self.file_path)

        # Validate schema
        self._validate_schema(df)

        # Parse timestamps
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date.astype(str)

        # Filter by date range if specified
        if start_date or end_date:
            df = self._filter_by_date(df, start_date, end_date)

        # Ensure Prompt column exists (may be missing in some datasets)
        if "Prompt" not in df.columns:
            df["Prompt"] = ""

        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Dataset missing required columns: {missing}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )

        if df.empty:
            raise ValueError("Dataset is empty")

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Filter DataFrame by date range.

        Args:
            df: DataFrame to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered DataFrame
        """
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["Timestamp"] >= start_dt]

        if end_date:
            # Add one day to make end_date inclusive
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            df = df[df["Timestamp"] < end_dt]

        if df.empty:
            raise ValueError(
                f"No data found in date range: {start_date} to {end_date}"
            )

        return df

    def get_user_list(self, df: pd.DataFrame) -> list[str]:
        """Get list of unique users in dataset.

        Args:
            df: DataFrame

        Returns:
            List of unique user identifiers
        """
        return df["Sender"].unique().tolist()

    def get_date_range(self, df: pd.DataFrame) -> tuple[str, str]:
        """Get date range of dataset.

        Args:
            df: DataFrame

        Returns:
            Tuple of (start_date, end_date) as 'YYYY-MM-DD' strings
        """
        start = df["Timestamp"].min().strftime("%Y-%m-%d")
        end = df["Timestamp"].max().strftime("%Y-%m-%d")
        return start, end
