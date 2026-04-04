from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from services.tb_predict import predict_tb
from services.brain_predict import predict_brain
from services.skin_predict import predict_skin
from services.oct_predict import predict_oct

app = FastAPI()

# ✅ CORS fix — allows frontend (HTML file) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],        # Local Live Ser    ver origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Multi Disease API Running"}

@app.post("/predict/{disease}")
async def predict(disease: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if disease == "tb":
        return predict_tb(image)
    elif disease == "brain":
        return predict_brain(image)
    elif disease == "skin":
        return predict_skin(image)
    elif disease == "oct":
        return predict_oct(image)
    else:
        return {"error": "Invalid disease. Use: tb, brain, skin, oct"}