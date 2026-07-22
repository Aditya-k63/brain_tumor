# Brain Tumor MRI Classifier

A deep learning project that classifies brain MRI scans into 4 categories — **Glioma, Meningioma, No Tumor, and Pituitary** — using EfficientNetB0 with transfer learning. Built with FastAPI backend, GradCAM explainability, and a built-in web UI. Deployed on Render with Docker and GitHub Actions CI/CD.

---

## Results

| Class | Recall | Precision | F1-Score |
|---|---|---|---|
| Glioma | 83% | 85% | 84% |
| Meningioma | 98% | 96% | 97% |
| No Tumor | 100% | 98% | 99% |
| Pituitary | 98% | 99% | 98% |

**Overall Accuracy: 95%**

Glioma recall was the hardest to fix — went through 4 iterations to get it right. Details below.

---

## Live Demo

**Render:** https://brain-tumor-xw48.onrender.com

Upload any brain MRI scan (JPG/PNG) and get instant classification with confidence scores and GradCAM heatmap.

---

## Project Structure

```
brain_tumor/
├── main.py                 # FastAPI backend
├── app.py                  # Streamlit frontend (local use)
├── utils/
│   ├── predict.py          # Model loading, preprocessing, prediction
│   └── gradcam.py          # GradCAM visualization
├── static/
│   └── index.html          # Built-in web UI (served by FastAPI)
├── tests/
│   └── test_api.py         # 18 tests
├── model/
│   └── brain_tumor_final.keras
├── download_model.py       # HuggingFace model downloader
├── requirements.txt
├── Dockerfile
├── supervisord.conf
├── docker-compose.yml
└── .github/workflows/
    └── ci.yml              # build-check → docker-build → docker-push
```

---

## How to Run

### Option 1 — Docker (recommended)

```bash
docker-compose up --build
```

Open `http://localhost:8000` for the web UI.

### Option 2 — Local

```bash
pip install -r requirements.txt
python download_model.py

# Terminal 1 — FastAPI
python main.py

# Terminal 2 — Streamlit (optional, for local UI)
streamlit run app.py
```

---

## Tech Stack

- **Model:** EfficientNetB0 (transfer learning from ImageNet)
- **Backend:** FastAPI with async endpoints
- **Frontend:** Built-in HTML UI + Streamlit (local)
- **Explainability:** GradCAM heatmap overlays
- **Dataset:** Brain Tumor MRI Dataset (Kaggle) — 4 classes, ~7000 images
- **Deployment:** Render (free tier) + Docker + GitHub Actions CI/CD
- **Model Hosted:** HuggingFace Hub (`neuronsbyisshu/brain-tumor-model`)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | Model status |
| POST | `/predict` | Single image classification |
| POST | `/predict/batch` | Batch classification (max 10) |
| POST | `/predict/gradcam` | GradCAM heatmap overlay |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## Problems I Faced & How I Fixed Them

### Problem 1 — Glioma Recall Was Only 65%

After training, validation accuracy was 91%. But on the actual test set, glioma recall dropped to 65%. That means 35 out of every 100 glioma patients would be missed — completely unacceptable for a medical application.

The confusion matrix showed 96 glioma cases being classified as meningioma. Both tumors can look similar on MRI, so the model was confused.

**Fix:** Applied a custom glioma threshold. Instead of using argmax, if the model gives glioma even 10% probability, I classify it as glioma. This boosted recall from 65% to 83%.

---

### Problem 2 — Distribution Shift Between Training and Test Data

The training and test glioma images in this dataset come from different sources with different scan protocols and contrast levels. The model learned training-set patterns so well it couldn't generalize to test-set images.

**Fix:** Merged Training and Testing folders into one Combined folder, then let ImageDataGenerator randomly split 80/20. Both sets now have a mix of both sources.

```
Training glioma:  826 images
Testing glioma:   300 images
Combined glioma:  1126 images → 900 train / 226 val (random split)
```

Glioma recall jumped from 65% to 79% (default threshold) and 89% (threshold 0.10).

---

### Problem 3 — Threshold 0.10 Killed Meningioma Recall

Threshold 0.10 gave best glioma recall (89%) but meningioma recall collapsed to 64%. I ran a threshold sweep and found the sweet spot at 0.30 — both classes above 80%.

**Final threshold: 0.10** — after retraining with combined data and focal loss, this gave the best balance: glioma 83%, meningioma 98%.

---

### Problem 4 — GradCAM Was Not Focusing on Tumor

Heatmap was broad and diffuse — lighting up the entire brain instead of the tumor.

**Fix:**
- Changed target layer from `top_conv` to `top_activation`
- Applied ReLU to gradients before pooling (only keep positive gradients)
- Used `cv2.INTER_CUBIC` for smoother upscaling
- Added gamma correction (`np.power(heatmap, 0.7)`) to boost contrast

---

### Problem 5 — Keras 3 Model Won't Load in Docker

TensorFlow 2.16 ships with Keras 2 by default, but the model was saved with Keras 3. The `quantization_config` error kept crashing the container.

**Fix:** Rebuilt the model architecture in code (EfficientNetB0 + GlobalAveragePooling2D + Dense 256 + Dense 4), then loaded just the weights. Also added `tf-keras` as a fallback for loading `.keras` files.

---

### Problem 6 — Render Free Tier Blocks Port 8501

Streamlit runs on port 8501 but Render only exposes port 8000. The UI was unreachable.

**Fix:** Built a custom HTML frontend (`static/index.html`) served directly by FastAPI on port 8000. Dark theme, drag-and-drop upload, live predictions, GradCAM overlay — all in one page.

---

## Training Strategy

| Phase | What I Did | Why |
|---|---|---|
| Phase 1 | Frozen base, train head only | Let the new Dense layers learn first |
| Phase 2 | Unfreeze last 40 layers, low LR | Fine-tune EfficientNet features for MRI |
| Phase 3 | Focal loss + class weights | Fix glioma/meningioma recall imbalance |

---

## Docker Details

The Dockerfile uses a single-stage build with supervisord:
- **FastAPI** runs always on port 8000
- **Streamlit** runs conditionally (set `RUN_STREAMLIT=true` for local dev)
- Model is downloaded from HuggingFace during build
- Built-in HTML UI is served by FastAPI

```bash
# Local with both services
docker-compose up --build

# Render (API only, UI is built-in)
# Just deploy — supervisord handles it
```

---

## CI/CD Pipeline

GitHub Actions workflow (`ci.yml`) with 3 jobs:

1. **build-check** — runs tests, verifies model loads
2. **docker-build** — builds Docker image
3. **docker-push** — pushes to Docker Hub (`adsharma14/brain-tumor`)

Triggered on push to `main`.

---

## What I Learned

- Threshold tuning matters more than accuracy in medical applications
- Distribution shift is real — always check test set sources
- GradCAM is powerful but needs careful layer selection
- Docker + supervisord is a clean way to run multiple services in one container
- Render free tier is limited (port 8000 only) but works for demos
- Keras 2 vs Keras 3 serialization is a headache — weights-only saves are safer
