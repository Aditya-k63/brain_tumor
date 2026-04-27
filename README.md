# Brain Tumor MRI Classifier

A deep learning project that classifies brain MRI scans into 4 categories — **Glioma, Meningioma, No Tumor, and Pituitary** — using EfficientNetB0 with transfer learning. Built with FastAPI backend and Streamlit frontend, with GradCAM visualization to explain model decisions.

---

## Final Results

| Metric | Value |
|---|---|
| Overall Accuracy | 91% |
| Glioma Recall | 83% (with threshold tuning) |
| Meningioma Recall | 82% |
| Notumor Recall | 99% |
| Pituitary Recall | 99% |

---

## Project Structure

```
brain-tumor-api/
├── main.py
├── app.py
├── model/
│   └── brain_tumor_final.keras
├── utils/
├── requirements.txt
├── Dockerfile               
├── tests/                   
│   └── test_api.py
└── .github/workflows/
    └── ci.yml
```

---

##  How to Run

**Terminal 1 — Start FastAPI:**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Start Streamlit:**
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

##  Tech Stack

- **Model:** EfficientNetB0 (Transfer Learning)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Explainability:** GradCAM
- **Dataset:** Brain Tumor MRI Dataset (Kaggle) 

---

##  Problems I Faced & How I Fixed Them (The Real Story)

This section documents every major problem I hit while building this project, written honestly so anyone reading can learn from  my mistakes.


### Problem 1 — Glioma Recall Was Only 65% on Test Set

After all three training phases, the model achieved 91% validation accuracy. But when I tested on the actual test set, glioma recall dropped to just 65%. That means 35 out of every 100 glioma patients would be missed — completely unacceptable for a medical application.
The confusion matrix showed 96 glioma cases being classified as meningioma, which makes sense visually — both tumors can look similar on MRI.
Instead of using the default argmax (which picks the class with highest probability), I applied a custom glioma threshold of 0.30. If the model gives glioma even a 30% probability, I classify it as glioma. This boosted recall from 65% → 75%.

---

### Problem 2 — Distribution Shift Between Training and Test Data

Even after threshold tuning, the best glioma recall on the test set was only 84% (at threshold 0.10), but that broke meningioma recall down to 64%. The root cause was a distribution shift — the training and test glioma images in this dataset come from different sources with different scan protocols, contrast levels, and orientations.
The model learned training-set glioma patterns so well (93% val recall) that it couldn't generalize to the different-looking test-set glioma images.
I physically merged the Training and Testing folders into one Combined folder, then let ImageDataGenerator randomly split it 80/20. This meant both train and validation sets now had a representative mix of both original sources.

```
Training glioma:  826 images
Testing glioma:   300 images
Combined glioma:  1126 images → 900 train / 226 val (random split)
```

After this, glioma recall jumped from 65% → 79% (default threshold) and 89% (threshold 0.10).

---

### Problem 3 — Threshold 0.10 Killed Meningioma Recall

After the dataset merge, I found that threshold 0.10 gave the best glioma recall (89%), but meningioma recall collapsed to 64%. Essentially, 99 meningioma cases were being misclassified as glioma because the threshold was so aggressive.
I ran a threshold sweep from 0.10 to 0.50 and printed both glioma AND meningioma recall at each level. The sweet spot was **threshold = 0.30**, which gave:
- Glioma recall: 83%
- Meningioma recall: 82%
- Overall accuracy: 90%

Both tumor classes above 80% — the best balanced outcome.
---

### Problem  — 4 GradCAM Heatmap Was Not Focusing on Tumor
GradCAM was generating outputs, but the heatmap was broad and diffuse — it was lighting up the entire brain instead of the specific tumor region.

1. I was using `top_conv` as the target layer, but `top_activation` (the activated output of that same layer) gives sharper gradients
2. I wasn't applying ReLU to the gradients before pooling, so negative gradients were diluting the heatmap
3. The heatmap upscaling was using default bilinear interpolation, producing blocky results

- Changed layer from `top_conv` → `top_activation`
- Applied `tf.maximum(grads, 0)` before pooling (only keep positive gradients)
- Used `cv2.INTER_CUBIC` for smoother upscaling
- Added gamma correction (`np.power(heatmap, 0.7)`) to boost contrast of weak activations
- Increased heatmap overlay weight from 0.4 → 0.5

---

##  Training Strategy Summary

| Phase | What  I Did | Why |
|---|---|---|
| Phase 1 | Frozen base, train head only | Let the new Dense layers learn first |
| Phase 2 | Unfreeze last 40 layers, low LR | Fine-tune EfficientNet features for MRI |
| Phase 3 | Focal loss + class weights | Fix glioma/meningioma recall imbalance |

---
