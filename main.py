"""
=====================================================
 API FastAPI — Detección de Plagas en Plantas
 Modelo: EfficientNetB3 (TFLite)
 Deploy: Render.com
=====================================================
"""

import os
import io
import json
import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

# TFLite runtime — más liviano que TF completo
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback a TF completo si tflite_runtime no está disponible
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Configuración ─────────────────────────────────────
MODEL_PATH  = Path("model/plant_disease.tflite")
LABELS_PATH = Path("model/class_names.json")
IMG_SIZE    = 300
TOP_K       = 5

# Estado global del modelo (se carga una sola vez al iniciar)
state = {}


# ── Lifespan: carga el modelo al arrancar el servidor ─
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Cargando modelo TFLite...")
    t0 = time.time()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Etiquetas no encontradas: {LABELS_PATH}")

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    state["interpreter"]     = interpreter
    state["input_details"]   = interpreter.get_input_details()
    state["output_details"]  = interpreter.get_output_details()
    state["input_shape"]     = interpreter.get_input_details()[0]["shape"]

    with open(LABELS_PATH, encoding="utf-8") as f:
        state["labels"] = json.load(f)

    elapsed = time.time() - t0
    print(f"[INFO] Modelo cargado en {elapsed:.2f}s")
    print(f"[INFO] Input shape: {state['input_shape']}")
    print(f"[INFO] Clases: {len(state['labels'])}")

    yield   # servidor corriendo

    state.clear()
    print("[INFO] Modelo descargado")


# ── App ───────────────────────────────────────────────
app = FastAPI(
    title="🌿 Plant Disease Detector",
    description="Detecta enfermedades en plantas usando EfficientNetB3 + TFLite",
    version="1.0.0",
    lifespan=lifespan
)


# ── Preprocesamiento (igual que preprocess_input de EfficientNet) ──
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # EfficientNet preprocess_input: escala a [-1, 1]
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def predecir(img_array: np.ndarray) -> list:
    interp = state["interpreter"]
    inp    = state["input_details"]
    out    = state["output_details"]
    labels = state["labels"]

    interp.set_tensor(inp[0]["index"], img_array)
    interp.invoke()
    probs = interp.get_tensor(out[0]["index"])[0]

    top_idxs = np.argsort(probs)[::-1][:TOP_K]
    return [
        {
            "rank":       int(i + 1),
            "clase":      labels.get(str(idx), f"clase_{idx}"),
            "confianza":  round(float(probs[idx]) * 100, 2),
            "saludable":  "healthy" in labels.get(str(idx), "").lower()
        }
        for i, idx in enumerate(top_idxs)
    ]


# ════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    """Página de bienvenida con formulario de prueba."""
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌿 Plant Disease Detector</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0f1117; color: #e2e8f0; min-height: 100vh;
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; padding: 2rem;
            }
            .card {
                background: #1a1d27; border: 1px solid #2d3748;
                border-radius: 16px; padding: 2.5rem;
                max-width: 520px; width: 100%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            }
            h1 { font-size: 1.8rem; margin-bottom: 0.4rem; }
            .sub { color: #718096; margin-bottom: 2rem; font-size: 0.9rem; }
            .drop-zone {
                border: 2px dashed #4a5568; border-radius: 12px;
                padding: 2.5rem; text-align: center; cursor: pointer;
                transition: all 0.2s; margin-bottom: 1.2rem;
                background: #12151f;
            }
            .drop-zone:hover, .drop-zone.drag { border-color: #48bb78; background: #1a2a1e; }
            .drop-zone p { color: #718096; font-size: 0.9rem; margin-top: 0.5rem; }
            #preview {
                width: 100%; max-height: 240px; object-fit: contain;
                border-radius: 8px; display: none; margin-bottom: 1rem;
            }
            button {
                width: 100%; padding: 0.9rem;
                background: #48bb78; color: #fff; border: none;
                border-radius: 10px; font-size: 1rem; font-weight: 600;
                cursor: pointer; transition: background 0.2s;
            }
            button:hover { background: #38a169; }
            button:disabled { background: #2d3748; cursor: not-allowed; }
            #result { margin-top: 1.5rem; display: none; }
            .badge {
                display: inline-block; padding: 0.3rem 0.9rem;
                border-radius: 99px; font-weight: 700; font-size: 0.85rem;
                margin-bottom: 1rem;
            }
            .healthy { background: #1c4532; color: #68d391; }
            .sick    { background: #4a1515; color: #fc8181; }
            .bar-row { margin-bottom: 0.6rem; }
            .bar-label { font-size: 0.78rem; color: #a0aec0; margin-bottom: 2px;
                         white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .bar-bg { background: #2d3748; border-radius: 99px; height: 8px; }
            .bar-fill { height: 8px; border-radius: 99px; transition: width 0.6s ease; }
            .bar-conf { font-size: 0.75rem; color: #718096; text-align: right; }
            .error { background: #4a1515; color: #fc8181; padding: 1rem;
                     border-radius: 8px; font-size: 0.9rem; }
            .spinner { border: 3px solid #2d3748; border-top-color: #48bb78;
                       border-radius: 50%; width: 24px; height: 24px;
                       animation: spin 0.8s linear infinite; margin: 0 auto; }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌿 Plant Disease Detector</h1>
            <p class="sub">EfficientNetB3 · 38 clases · PlantVillage dataset</p>

            <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <span style="font-size:2.5rem">🍃</span>
                <p>Haz clic o arrastra una foto de hoja</p>
                <p style="font-size:0.8rem;margin-top:4px">JPG, PNG · max 10 MB</p>
            </div>
            <img id="preview" alt="Preview">
            <input type="file" id="fileInput" accept="image/*" style="display:none">

            <button id="btn" onclick="analizar()" disabled>Analizar planta</button>
            <div id="result"></div>
        </div>

        <script>
        const fileInput = document.getElementById('fileInput');
        const preview   = document.getElementById('preview');
        const btn       = document.getElementById('btn');
        const dropZone  = document.getElementById('dropZone');
        let selectedFile = null;

        fileInput.addEventListener('change', e => showPreview(e.target.files[0]));

        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault(); dropZone.classList.remove('drag');
            showPreview(e.dataTransfer.files[0]);
        });

        function showPreview(file) {
            if (!file) return;
            selectedFile = file;
            const url = URL.createObjectURL(file);
            preview.src = url; preview.style.display = 'block';
            btn.disabled = false;
            document.getElementById('result').style.display = 'none';
        }

        async function analizar() {
            if (!selectedFile) return;
            btn.disabled = true;
            const res = document.getElementById('result');
            res.style.display = 'block';
            res.innerHTML = '<div class="spinner"></div>';

            const form = new FormData();
            form.append('file', selectedFile);

            try {
                const r = await fetch('/predict', { method: 'POST', body: form });
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail || 'Error en la predicción');

                const top = d.predicciones[0];
                const colors = ['#48bb78','#68d391','#9ae6b4','#c6f6d5','#e2e8f0'];
                const badgeClass = top.saludable ? 'healthy' : 'sick';
                const badgeText  = top.saludable ? '✅ SANA' : '🔴 ENFERMA';

                res.innerHTML = `
                    <span class="badge ${badgeClass}">${badgeText}</span>
                    ${d.predicciones.map((p, i) => `
                        <div class="bar-row">
                            <div class="bar-label">${p.clase.replace(/_/g,' ')}</div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width:${p.confianza}%;background:${colors[i]}"></div>
                            </div>
                            <div class="bar-conf">${p.confianza.toFixed(1)}%</div>
                        </div>
                    `).join('')}
                    <p style="font-size:0.75rem;color:#4a5568;margin-top:0.8rem">
                        ⏱ ${d.tiempo_ms} ms
                    </p>`;
            } catch(e) {
                res.innerHTML = `<div class="error">❌ ${e.message}</div>`;
            }
            btn.disabled = false;
        }
        </script>
    </body>
    </html>
    """


@app.post("/predict", tags=["Predicción"])
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y retorna el diagnóstico de la planta.
    
    - **file**: imagen JPG o PNG de una hoja de planta
    """
    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (JPG, PNG)")

    # Validar tamaño (max 10 MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Imagen demasiado grande (max 10 MB)")

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    t0 = time.time()
    img_array    = preprocess(img)
    predicciones = predecir(img_array)
    elapsed_ms   = round((time.time() - t0) * 1000, 1)

    return JSONResponse({
        "archivo":     file.filename,
        "predicciones": predicciones,
        "tiempo_ms":   elapsed_ms,
        "modelo":      "EfficientNetB3 TFLite (Fase 1 — 90.6% val_accuracy)"
    })


@app.get("/health", tags=["General"])
async def health():
    """Verifica que el servicio y el modelo están listos."""
    return {
        "status":  "ok",
        "modelo":  "cargado" if "interpreter" in state else "no cargado",
        "clases":  len(state.get("labels", {}))
    }


@app.get("/clases", tags=["General"])
async def listar_clases():
    """Lista todas las clases que el modelo puede detectar."""
    labels = state.get("labels", {})
    return {
        "total": len(labels),
        "clases": [
            {
                "id":        int(k),
                "nombre":    v.replace("_", " "),
                "saludable": "healthy" in v.lower()
            }
            for k, v in sorted(labels.items(), key=lambda x: int(x[0]))
        ]
    }
