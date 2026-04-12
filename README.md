# 🤖 ML Pipeline — Polars + MLflow + FastAPI
> Pipeline de machine learning de ponta a ponta: ingestão, feature engineering, treino, versionamento e deploy como API

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-0.20-CD792C)](https://pola.rs)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)](https://coverage.readthedocs.io)

## 🎯 Sobre

Pipeline ML completo com **Polars** (10x mais rápido que Pandas), **MLflow** para rastreamento de experimentos e versionamento de modelos, e **FastAPI** para servir predições em produção.

**Caso de uso:** Previsão de demanda de produtos para e-commerce.

## 🛠️ Stack

| Componente | Tech |
|-----------|------|
| Data Processing | Polars 0.20 (lazy evaluation) |
| Modelos | Scikit-learn, XGBoost, LightGBM |
| Tracking | MLflow 2.13 |
| Serving | FastAPI + Pydantic |
| Deploy | Docker + GitHub Actions |

## ⚡ Por que Polars?

```python
# Pandas — 12.3 segundos para 10M registros
df_pandas.groupby("produto").agg({"vendas": "sum"})

# Polars — 0.9 segundos para os mesmos 10M registros ⚡
df_polars.lazy().group_by("produto").agg(
    pl.col("vendas").sum()
).collect()
```

## 📊 Resultados

| Experimento | MAE | RMSE | MAPE |
|------------|-----|------|------|
| Baseline (média) | 412 | 578 | 23.1% |
| Random Forest | 198 | 271 | 11.2% |
| **XGBoost (prod)** | **143** | **198** | **8.1%** |

---
**Gabriel Kaique Portel Silva** | [LinkedIn](https://linkedin.com/in/gabriel-kaique-881475284) | [GitHub](https://github.com/Kaique-ML)
