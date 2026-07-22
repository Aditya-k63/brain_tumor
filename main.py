import os
import uuid
import logging
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List

from utils.predict import load_model, predict, CLASS_NAMES
from utils.gradcam import generate_gradcam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model/brain_tumor_final.weights.h5")
HF_REPO_ID = os.getenv("HF_REPO_ID", "neuronsbyisshu/brain-tumor-model")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.40"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting up — loading model...")

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}, running without model")
        model = None
    else:
        size = os.path.getsize(MODEL_PATH)
        logger.info(f"Model found — size: {size / 1024 / 1024:.1f} MB")
        try:
            model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            model = None

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="EfficientNetB0-based brain tumor classification with GradCAM explainability",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.time()

    response = await call_next(request)

    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({elapsed}ms)")
    response.headers["X-Request-ID"] = request_id
    return response


def validate_file(file: UploadFile, content: bytes):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Allowed: JPEG, PNG")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {len(content) / 1024 / 1024:.1f}MB (max {MAX_FILE_SIZE / 1024 / 1024:.0f}MB)")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(content))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")


@app.get("/")
def home():
    return {"message": "Brain Tumor MRI Classifier API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "loading",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "min_confidence": MIN_CONFIDENCE,
    }


@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    content = await file.read()
    validate_file(file, content)

    try:
        result = predict(model, content)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    if result["confidence"] < MIN_CONFIDENCE * 100:
        result["warning"] = f"Low confidence prediction ({result['confidence']}%). Treat with caution."
        result["below_threshold"] = True

    result["filename"] = file.filename
    result["model_version"] = "2.0.0"
    return result


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    for file in files:
        content = await file.read()
        try:
            validate_file(file, content)
            result = predict(model, content)
            result["filename"] = file.filename
            if result["confidence"] < MIN_CONFIDENCE * 100:
                result["warning"] = f"Low confidence ({result['confidence']}%)"
                result["below_threshold"] = True
            results.append(result)
        except HTTPException as e:
            results.append({"filename": file.filename, "error": e.detail})
        except Exception as e:
            logger.error(f"Batch prediction failed for {file.filename}: {e}")
            results.append({"filename": file.filename, "error": "Prediction failed"})

    return {"count": len(results), "results": results}


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    content = await file.read()
    validate_file(file, content)

    try:
        gradcam_bytes = generate_gradcam(model, content)
    except Exception as e:
        logger.error(f"GradCAM failed: {e}")
        raise HTTPException(status_code=500, detail="GradCAM generation failed")

    return Response(content=gradcam_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
