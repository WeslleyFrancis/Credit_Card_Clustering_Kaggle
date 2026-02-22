import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def drop_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    return df.drop(columns=[id_col])

def impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out[c].median())
    return out

def log1p_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = np.log1p(out[c])
    return out

def zscore_scale(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[cols])
    return pd.DataFrame(arr, columns=cols), scaler