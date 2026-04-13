"""Feature engineering com Polars (lazy evaluation)."""
import polars as pl

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .with_columns([
            # 1. Primeiro converte a string para data real
            pl.col("data").str.to_date("%Y-%m-%d"),
        ])
        .with_columns([
            # 2. Agora as operações de data e lags vão funcionar
            (pl.col("vendas").shift(7)).alias("vendas_lag7"),
            (pl.col("vendas").shift(30)).alias("vendas_lag30"),
            pl.col("data").dt.weekday().alias("dia_semana"),
            pl.col("data").dt.month().alias("mes"),
            pl.col("vendas").rolling_mean(window_size=7).alias("media_movel_7d"),
        ])
        .drop_nulls()
        .collect()
    )