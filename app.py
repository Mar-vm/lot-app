from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="Fake Tomato Pest AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message":"Tomato IA running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    await file.read()
    return {
        "success": True,
        "crop": "Tomate",
        "prediction": {
            "plaga_detectada": "Tizón temprano",
            "probabilidad": round(random.uniform(88,95),2),
            "severidad": "Media",
            "riesgo": "Alto",
            "recomendaciones": [
                "Aplicar fungicida preventivo",
                "Retirar hojas afectadas",
                "Monitorear humedad del cultivo",
                "Realizar inspección en 48 horas"
            ]
        },
        "modelo": "TomatoPestNet-v1"
    }
