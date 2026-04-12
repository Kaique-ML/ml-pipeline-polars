"""FastAPI serving para predições do modelo."""
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="ML Pipeline — Serving")
model = mlflow.sklearn.load_model(os.getenv("MODEL_URI", "models:/demand-forecast/Production"))


class PredictRequest(BaseModel):
    vendas_lag7: float
    vendas_lag30: float
    dia_semana: int
    mes: int
    media_movel_7d: float


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    pred = model.predict(df)[0]
    return {"prediction": round(float(pred), 2)}


@app.get("/health")
def health():
    return {"status": "ok"}
