from .preprocessing import (
    drop_id,
    impute_median,
    log1p_transform,
    zscore_scale,
    validate_no_missing,
)
from . import config


def build_processed_datasets(df_raw, config):
    df = drop_id(df_raw, config.ID_COL)

    # imputação dinâmica: todas as colunas numéricas com nulos
    df = impute_median(df)

    # validação após imputação
    validate_no_missing(df, stage_name="apos_imputacao")

    df = log1p_transform(df, config.LOG1P_COLS)

    # validação extra, por segurança
    validate_no_missing(df, stage_name="apos_log1p")

    full_cols = df.columns.tolist()

    out = {}
    out["risk"], scaler_risk = zscore_scale(df, config.RISK_COLS)
    out["credit"], scaler_credit = zscore_scale(df, config.CREDIT_BEHAVIOR_COLS)
    out["consumption"], scaler_cons = zscore_scale(df, config.CONSUMPTION_COLS)
    out["full"], scaler_full = zscore_scale(df, full_cols)

    # validação final dos datasets processados
    for name, dataset in out.items():
        validate_no_missing(dataset, stage_name=f"dataset_processado_{name}")

    scalers = {
        "risk": scaler_risk,
        "credit": scaler_credit,
        "consumption": scaler_cons,
        "full": scaler_full,
    }

    return out, scalers