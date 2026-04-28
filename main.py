import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

from utils.predict import load_model, predict
from utils.gradcam import generate_gradcam

MODEL_PATH = "model/brain_tumor_final.keras"
HF_REPO_ID = "Aditya-k63/brain-tumor-model"  # your HF repo
model = None

def ensure_model():
    """Download model from HuggingFace if not present locally."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        print("Model not found locally. Downloading from HuggingFace...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="brain_tumor_final.keras",
            local_dir="model"
        )
        print("Model downloaded successfully!")
    else:
        print("Model found locally, skipping download.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    ensure_model()                 
    model = load_model(MODEL_PATH)  
    print("Model loaded successfully")
    yield
    print("Shutting down")

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="EfficientNetB0 based brain tumor classification with GradCAM",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
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

@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)