"""Leitura e validação de dados com Polars."""
import polars as pl
from pathlib import Path


def load_data(path: str) -> pl.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif p.suffix == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Formato não suportado: {p.suffix}")

    assert len(df) > 0, "Dataset vazio!"
    print(f"✅ Dados carregados: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df
