from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="Tomato Pest AI Multi-Class")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base simulada de enfermedades/plagas
PLAGAS = [
    {
        "nombre":"Tizón temprano",
        "severidad":["Media","Alta"],
        "riesgo":"Alto",
        "recomendaciones":[
            "Aplicar fungicida preventivo",
            "Retirar hojas afectadas",
            "Reducir humedad en el cultivo",
            "Monitoreo cada 48 horas"
        ]
    },
    {
        "nombre":"Tizón tardío",
        "severidad":["Alta","Crítica"],
        "riesgo":"Muy alto",
        "recomendaciones":[
            "Aplicar fungicida sistémico",
            "Eliminar plantas severamente afectadas",
            "Evitar riego por aspersión",
            "Aislar zona infectada"
        ]
    },
    {
        "nombre":"Mildiu",
        "severidad":["Media"],
        "riesgo":"Medio",
        "recomendaciones":[
            "Mejorar ventilación",
            "Reducir exceso de humedad",
            "Aplicar tratamiento preventivo"
        ]
    },
    {
        "nombre":"Mosca blanca",
        "severidad":["Media","Alta"],
        "riesgo":"Alto",
        "recomendaciones":[
            "Usar control biológico",
            "Aplicar jabón potásico",
            "Colocar trampas amarillas"
        ]
    },
    {
        "nombre":"Ácaros",
        "severidad":["Baja","Media"],
        "riesgo":"Medio",
        "recomendaciones":[
            "Aplicar acaricida",
            "Incrementar monitoreo",
            "Retirar hojas dañadas"
        ]
    },
    {
        "nombre":"Septoria",
        "severidad":["Media","Alta"],
        "riesgo":"Alto",
        "recomendaciones":[
            "Aplicar fungicida foliar",
            "Eliminar residuos infectados",
            "Rotación de cultivos"
        ]
    }
]

@app.get("/")
def root():
    return {
        "message":"Tomato Multi-Class Pest AI running",
        "clases_soportadas": len(PLAGAS)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    await file.read()

    # Simula top-3 predicciones del modelo
    detecciones = random.sample(PLAGAS, 3)

    probabilidades = sorted(
        [round(random.uniform(82,97),2) for _ in range(3)],
        reverse=True
    )

    resultados = []

    for i,plaga in enumerate(detecciones):
        resultados.append({
            "plaga": plaga["nombre"],
            "probabilidad": probabilidades[i],
            "severidad": random.choice(plaga["severidad"]),
            "riesgo": plaga["riesgo"],
            "recomendaciones": plaga["recomendaciones"]
        })

    return {
        "success":True,
        "crop":"Tomate",
        "diagnostico_principal": resultados[0],
        "top_predicciones": resultados,
        "numero_clases_modelo": len(PLAGAS),
        "estado_cultivo":"Requiere atención",
        "modelo":"TomatoPestNet-v2 MultiDisease"
    }
