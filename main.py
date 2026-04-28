import os
import uvicorn
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

from utils.predict import load_model, predict
from utils.gradcam import generate_gradcam

MODEL_PATH = "model/brain_tumor_final.keras"
HF_REPO_ID = "neurobyisshu/brain-tumor-model"

model = None


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Hugging Face...")

        model_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="brain_tumor_final.keras",
            local_dir="model"
        )
        return model_file
    return MODEL_PATH
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model_path = ensure_model()
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error in loading model: {e}")
        model = None

    yield
    print("Shutting down")
app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="EfficientNetB0 based brain tumor classification with GradCAM",
    version="1.0.0",
    lifespan=lifespan
)
@app.get("/")
def home():
    return {"message": "Brain Tumor API is running"}
@app.get("/health")
def health():
    return {
        "status": "ok" if model else "loading",
        "model_loaded": model is not None
    }

## predicting teh end points 

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG/PNG images accepted"
        )
    image_bytes = await file.read()

    try:
        result = predict(model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"filename": file.filename, **result}

# GradCAM endpoint

@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG/PNG images accepted"
        )

    image_bytes = await file.read()

    try:
        gradcam_bytes = generate_gradcam(model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=gradcam_bytes, media_type="image/png")

# Local run (for testing only)

if __name__ == "__main__":
    port=int(os.environ.get("PORT",8000))
    uvicorn.run("main:app", host="0.0.0.0",port=port)