import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLIOMA_THRESHOLD = 0.30

def load_model(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        safe_mode=False
    )

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image_bytes: bytes) -> dict:
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array)[0]

    # apply glioma threshold
    if preds[0] >= GLIOMA_THRESHOLD:
        predicted_class = 0
    else:
        predicted_class = int(np.argmax(preds))

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": round(float(preds[predicted_class]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(4)
        }
    }