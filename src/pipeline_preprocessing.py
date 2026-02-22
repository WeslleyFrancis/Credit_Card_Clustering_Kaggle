from .preprocessing import drop_id, impute_median, log1p_transform, zscore_scale
from . import config

def build_processed_datasets(df_raw, config):
    df = drop_id(df_raw, config.ID_COL)
    df = impute_median(df, config.IMPUTE_MEDIAN_COLS)
    df = log1p_transform(df, config.LOG1P_COLS)

    full_cols = df.columns.tolist()

    out = {}
    out["risk"], scaler_risk = zscore_scale(df, config.RISK_COLS)
    out["credit"], scaler_credit = zscore_scale(df, config.CREDIT_BEHAVIOR_COLS)
    out["consumption"], scaler_cons = zscore_scale(df, config.CONSUMPTION_COLS)
    out["full"], scaler_full = zscore_scale(df, full_cols)

    scalers = {"risk": scaler_risk, "credit": scaler_credit, "consumption": scaler_cons, "full": scaler_full}
    return out, scalers