from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from services.tb_predict import predict_tb
from services.brain_predict import predict_brain
from services.skin_predict import predict_skin
from services.oct_predict import predict_oct
from services.federated_info import get_federated_info, get_federated_status

app = FastAPI(title="Multi Disease Detection API")

# ─────────────────────────────────────────────
# CORS — allows the frontend HTML file (opened
# via Live Server or directly) to call this API.
# During development we allow all origins (*).
# Tighten this to a specific origin before deployment.
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Dev-mode: open to all origins
    allow_credentials=False,       # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_DISEASES = {"tb", "brain", "skin", "oct"}

@app.get("/")
def home():
    return {"status": "Multi Disease Detection API is running", "endpoints": ["/predict/tb", "/predict/brain", "/predict/skin", "/predict/oct"]}

@app.get("/health")
def health():
    """Quick health-check — useful for confirming backend is up before frontend tests."""
    return {"status": "ok"}

@app.get("/federated/info")
def federated_info():
    """Get metadata about the federated learning process."""
    return get_federated_info()

@app.get("/federated/status")
def federated_status():
    """Get the current global model hashes to verify FL updates."""
    return get_federated_status()

@app.post("/predict/{disease}")
async def predict(disease: str, file: UploadFile = File(...)):
    if disease not in VALID_DISEASES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid disease '{disease}'. Valid options: {sorted(VALID_DISEASES)}"
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Make sure you are uploading a valid image file.")

    try:
        if disease == "tb":
            return predict_tb(image)
        elif disease == "brain":
            return predict_brain(image)
        elif disease == "skin":
            return predict_skin(image)
        elif disease == "oct":
            return predict_oct(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")