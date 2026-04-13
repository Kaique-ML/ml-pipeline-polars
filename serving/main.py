"""FastAPI serving para predições do modelo."""
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="ML Pipeline — Serving")

# --- CONFIGURAÇÃO DO MODELO ---
# Substitua o ID abaixo pelo Run ID que aparece no seu MLflow (ex: "76a8b2c3...")
RUN_ID = "a4d1304051834b7db3d9f530a86d87f6"

# Definimos o endereço do servidor MLflow (Docker)
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Carregamos o modelo direto da pasta de experimentos (Runs)
model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)
# ------------------------------

class PredictRequest(BaseModel):
    vendas_lag7: float
    vendas_lag30: float
    dia_semana: int
    mes: int
    media_movel_7d: float

@app.post("/predict")
def predict(req: PredictRequest):
    # Converte o request para DataFrame (o modelo espera o mesmo formato do treino)
    df = pd.DataFrame([req.model_dump()])
    pred = model.predict(df)[0]
    return {"prediction": round(float(pred), 2)}

@app.get("/health")
def health():
    return {"status": "ok"}