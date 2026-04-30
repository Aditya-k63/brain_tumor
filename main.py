import os
import traceback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

from utils.predict import load_model, predict
from utils.gradcam import generate_gradcam

MODEL_PATH = "model/brain_tumor_final_fixed.h5"
HF_REPO_ID = "neuronsbyisshu/brain-tumor-model"

model = None


def ensure_model():
    print("DEBUG: Checking model path...")

    if not os.path.exists(MODEL_PATH):
        print(f" Model NOT FOUND at {MODEL_PATH}")
        print("Listing /app directory:")
        print(os.listdir("/app"))
        return None

    print(f" Model FOUND at {MODEL_PATH}")
    print("Listing model directory:")
    print(os.listdir("model"))

    return MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print("\n=== STARTUP DEBUG ===")

        # STEP 1: Check model existence
        model_path = ensure_model()

        if model_path is None:
            print(" Model path is None — skipping load")
            model = None
        else:
            print(f" Model path: {model_path}")

            # Extra debug
            if not os.path.exists(model_path):
                print(" Model file DOES NOT exist!")
                print(" /app contents:", os.listdir("/app"))
                if os.path.exists("model"):
                    print("model folder:", os.listdir("model"))
                model = None
            else:
                size = os.path.getsize(model_path)
                print(f" Model exists — size: {size} bytes")

                print(" Loading model into memory...")

                try:
                    model = load_model(model_path)
                    print(" MODEL LOADED SUCCESSFULLY")
                except Exception as load_error:
                    print(" MODEL LOAD FAILED:")
                    traceback.print_exc()
                    model = None

    except Exception as e:
        print(" STARTUP FAILED:")
        traceback.print_exc()
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
        "status": "ok" if model is not None else "loading",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images accepted")

    image_bytes = await file.read()
    try:
        result = predict(model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"filename": file.filename, **result}


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images accepted")

    image_bytes = await file.read()
    try:
        gradcam_bytes = generate_gradcam(model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=gradcam_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)