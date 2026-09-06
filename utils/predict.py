import numpy as np
from keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf
import io
import math


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLIOMA_THRESHOLD = 0.10

# OOD detection thresholds — tuned to WARN not reject
MAX_PROB_THRESHOLD = 0.35
ENTROPY_THRESHOLD = 1.2

NUM_CLASSES = len(CLASS_NAMES)
MAX_ENTROPY = math.log(NUM_CLASSES)


def build_model():
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)
    return model


def load_model(model_path: str):
    if model_path.endswith(".weights.h5"):
        model = build_model()
        model.load_weights(model_path)
        return model

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    try:
        import tf_keras
        return tf_keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    model = build_model()
    model.load_weights(model_path)
    return model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def _is_valid_image(image_bytes: bytes) -> dict:
    """Only reject clearly invalid inputs (color photos, corrupt images).

    Does NOT reject valid brain MRIs based on shape/contrast heuristics.
    Real brain MRIs come in many shapes, contrasts, and formats.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"ok": False, "reason": "Cannot read image"}

    img_array = np.array(img, dtype=np.float32)

    # Brain MRIs are grayscale (not color photos).
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    color_diff = (np.std(r - g) + np.std(g - b)) / 2
    if color_diff > 30:
        return {"ok": False, "reason": "Image has strong color — brain MRIs are grayscale"}

    # Reject very low contrast or near-blank images.
    gray = np.mean(img_array, axis=2)
    std = float(np.std(gray))
    if std < 5:
        return {"ok": False, "reason": "Image has very low contrast"}

    mean = float(np.mean(gray))
    if mean < 5 or mean > 250:
        return {"ok": False, "reason": "Image is almost entirely black or white"}

    return {"ok": True, "reason": ""}


def _compute_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def predict(model, image_bytes: bytes) -> dict:
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array, verbose=0)[0]

    # --- Class prediction with glioma boost (only when competitive) ---
    argmax_class = int(np.argmax(preds))
    if (
        preds[0] >= GLIOMA_THRESHOLD
        and preds[0] >= preds[argmax_class] * 0.5
    ):
        predicted_class = 0
    else:
        predicted_class = argmax_class

    max_prob = float(np.max(preds))
    entropy = _compute_entropy(preds)

    # OOD detection is a WARNING, not a rejection.
    # The model always returns a real class, but warns if uncertain.
    is_ood = (
        max_prob < MAX_PROB_THRESHOLD
        or entropy > ENTROPY_THRESHOLD
    )

    result = {
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": round(float(preds[predicted_class]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(NUM_CLASSES)
        },
        "is_ood": is_ood,
        "max_probability": round(max_prob * 100, 2),
        "entropy": round(entropy, 4),
    }

    if is_ood:
        result["warning"] = (
            "Low confidence prediction — this image may not be a brain MRI. "
            "Please consult a medical professional for a definitive diagnosis."
        )

    return result


def validate_image(image_bytes: bytes) -> dict:
    """Public wrapper to check whether an image is a valid input (not corrupt, not color photo).

    Returns {"ok": bool, "reason": str}.
    Does NOT reject valid brain MRIs.
    """
    return _is_valid_image(image_bytes)
