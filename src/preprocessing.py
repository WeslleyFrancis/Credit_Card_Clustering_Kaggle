import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    return df.drop(columns=[id_col])


def get_numeric_columns_with_na(df: pd.DataFrame) -> list[str]:
    """
    Retorna a lista de colunas numéricas que possuem pelo menos 1 valor nulo.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_with_na = [col for col in numeric_cols if df[col].isna().any()]
    return cols_with_na


def impute_median(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Imputa mediana nas colunas informadas.
    Se cols=None, detecta automaticamente todas as colunas numéricas com nulos.
    """
    out = df.copy()

    if cols is None:
        cols = get_numeric_columns_with_na(out)

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
    return pd.DataFrame(arr, columns=cols, index=df.index), scaler


def validate_no_missing(df: pd.DataFrame, stage_name: str = "dataframe") -> None:
    """
    Lança erro se ainda houver valores nulos após o tratamento.
    """
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]

    if not na_counts.empty:
        raise ValueError(
            f"Ainda existem nulos após a etapa '{stage_name}'.\n"
            f"Colunas com nulos:\n{na_counts.to_string()}"
        )