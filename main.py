
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager

from utils.predict import load_model, predict
from utils.gradcam import generate_gradcam

MODEL_PATH = "model/brain_tumor_final.keras"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
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

    return {
        "filename": file.filename,
        **result
    }

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

    return Response(
        content=gradcam_bytes,
        media_type="image/png"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)