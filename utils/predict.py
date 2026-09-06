import numpy as np
from keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf
import io
import math


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLIOMA_THRESHOLD = 0.10

# OOD detection thresholds
MAX_PROB_THRESHOLD = 0.50
ENTROPY_THRESHOLD = 0.8
ENERGY_THRESHOLD = -2.0

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


def _is_brain_mri(image_bytes: bytes) -> dict:
    """Heuristic check whether an image looks like a brain MRI.

    Returns {"ok": bool, "reason": str}.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"ok": False, "reason": "Cannot read image"}

    img_array = np.array(img, dtype=np.float32)
    gray = np.mean(img_array, axis=2)

    # Brain MRIs are mostly grayscale — channels should be nearly equal.
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    color_diff = (np.std(r - g) + np.std(g - b)) / 2
    if color_diff > 25:
        return {"ok": False, "reason": "Image has strong color — brain MRIs are grayscale"}

    # Standard deviation should be in a reasonable range for medical images.
    std = float(np.std(gray))
    if std < 10:
        return {"ok": False, "reason": "Image has very low contrast"}
    if std > 100:
        return {"ok": False, "reason": "Image has abnormally high contrast"}

    # Extreme mean brightness (nearly all black or all white).
    mean = float(np.mean(gray))
    if mean < 10 or mean > 245:
        return {"ok": False, "reason": "Image is almost entirely black or white"}

    return {"ok": True, "reason": ""}


def _compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    clipped = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _compute_energy(probs: np.ndarray) -> float:
    """Energy score: higher magnitude = more OOD."""
    return float(np.sum(np.exp(probs)))


def predict(model, image_bytes: bytes) -> dict:
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array, verbose=0)[0]

    # --- OOD detection ---
    max_prob = float(np.max(preds))
    entropy = _compute_entropy(preds)
    energy = _compute_energy(preds)

    is_uncertain = (
        max_prob < MAX_PROB_THRESHOLD
        or entropy > ENTROPY_THRESHOLD
        or energy < ENERGY_THRESHOLD
    )

    # --- class prediction with glioma boost ---
    argmax_class = int(np.argmax(preds))

    if (
        preds[0] >= GLIOMA_THRESHOLD
        and preds[0] >= preds[argmax_class] * 0.5
    ):
        predicted_class = 0
    else:
        predicted_class = argmax_class

    result = {
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": round(float(preds[predicted_class]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(NUM_CLASSES)
        },
        "is_ood": is_uncertain,
        "max_probability": round(max_prob * 100, 2),
        "entropy": round(entropy, 4),
    }

    if is_uncertain:
        result["warning"] = (
            "Low-confidence prediction — this image may not be a brain MRI. "
            "Results on non-MRI images are unreliable."
        )

    return result


def validate_image(image_bytes: bytes) -> dict:
    """Public wrapper to check whether an image looks like a brain MRI.

    Returns {"ok": bool, "reason": str}.
    """
    return _is_brain_mri(image_bytes)
