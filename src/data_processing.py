import numpy as np
import pandas as pd


def load_dataset(path: str = "data/data.xlsx") -> pd.DataFrame:
    """
    Load and preprocess the dataset to match the original notebook behavior.

    - Read Excel file
    - Parse 'Time' column with format 'dd.mm.yyyy. HH:MM:SS'
    - Drop rows with invalid timestamps
    - Sort by time and set as index
    - Keep only ['SM', 'P', 'T'] as float
    - Print date range, row count and basic statistics
    """
    df = pd.read_excel(path, engine="openpyxl")

    # Parse timestamps (format: "dd.mm.yyyy. HH:MM:SS")
    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%d.%m.%Y. %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["Time"]).sort_values("Time").set_index("Time")

    # Keep relevant columns and ensure float dtype
    df = df[["SM", "P", "T"]].astype(float)

    # Match notebook-style diagnostic output
    print("Date range:", df.index.min(), "→", df.index.max())
    print("Total rows:", len(df))
    print(df.describe())

    return df


def make_lag_matrix(series, lags):
    """
    Create lag matrix for univariate time series.
    """
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def make_narx_matrix(y, X_exog, lags_y, lags_x):
    """
    Create lag matrix for NARX model.
    """
    X, target = [], []

    for i in range(max(lags_y, lags_x), len(y)):
        y_lags = y[i - lags_y:i]
        x_lags = X_exog[i - lags_x:i].flatten()
        X.append(np.concatenate([y_lags, x_lags]))
        target.append(y[i])

    return np.array(X), np.array(target)