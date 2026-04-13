"""Treino do modelo com MLflow tracking."""
import argparse
import mlflow
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pipeline.ingest import load_data
from pipeline.features import build_features
import numpy as np


def train(data_path: str, model_type: str = "xgboost", experiment: str = "demand-forecast"):
    mlflow.set_experiment(experiment)

    df = load_data(data_path)
    df = build_features(df)

    # Usar pandas para sklearn
    pdf = df.to_pandas()
    feature_cols = [c for c in pdf.columns if c not in ["data", "vendas"]]
    X, y = pdf[feature_cols], pdf["vendas"]
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with mlflow.start_run(run_name=f"{model_type}-run"):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = (abs((y_test - preds) / y_test)).mean() * 100

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        mlflow.sklearn.log_model(
            sk_model=pipe, 
            artifact_path="model", 
            registered_model_name="demand-forecast"
        )
        print(f"MAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.1f}%")
    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--experiment", default="demand-forecast")
    args = parser.parse_args()
    train(args.data, args.model, args.experiment)
