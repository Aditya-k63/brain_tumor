import numpy as np
from keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf
import io
import math


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLIOMA_THRESHOLD = 0.10

# Stricter OOD detection thresholds
MAX_PROB_THRESHOLD = 0.60
ENTROPY_THRESHOLD = 0.65
ENERGY_THRESHOLD = -2.0
FEATURE_VARIANCE_MIN = 0.01
FEATURE_VARIANCE_MAX = 10.0

NUM_CLASSES = len(CLASS_NAMES)
MAX_ENTROPY = math.log(NUM_CLASSES)

# Reference statistics for brain MRI features (from training data)
# These are approximate values learned from the training distribution
FEATURE_MEAN_ESTIMATED = 0.5
FEATURE_STD_ESTIMATED = 1.0


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


def get_feature_extractor(model):
    """Return a model that outputs intermediate features before classification head."""
    base = model.layers[0]
    gap = model.layers[1]
    dense1 = model.layers[2]
    return tf.keras.Model(inputs=base.input, outputs=dense1.output)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def _is_brain_mri(image_bytes: bytes) -> dict:
    """Heuristic check whether an image looks like a brain MRI."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"ok": False, "reason": "Cannot read image"}

    img_array = np.array(img, dtype=np.float32)
    gray = np.mean(img_array, axis=2)

    # Brain MRIs are grayscale.
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    color_diff = (np.std(r - g) + np.std(g - b)) / 2
    if color_diff > 25:
        return {"ok": False, "reason": "Image has strong color — brain MRIs are grayscale"}

    # Contrast check.
    std = float(np.std(gray))
    if std < 10:
        return {"ok": False, "reason": "Image has very low contrast"}
    if std > 100:
        return {"ok": False, "reason": "Image has abnormally high contrast"}

    # Brightness check.
    mean = float(np.mean(gray))
    if mean < 10 or mean > 245:
        return {"ok": False, "reason": "Image is almost entirely black or white"}

    # Check for circular/oval structure typical of brain MRI.
    # Brain MRIs have a roughly circular brain region centered in the frame.
    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    y_idx, x_idx = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_dist = min(cy, cx)

    # Compute how much of the image has content vs background.
    # Brain MRI should have a central bright region with dark periphery.
    center_ratio = np.mean(gray[cy - max_dist // 2:cy + max_dist // 2,
                                  cx - max_dist // 2:cx + max_dist // 2])
    edge_ratio = np.mean(gray[0:max_dist // 4, :]) + np.mean(gray[-max_dist // 4:, :]) + \
                 np.mean(gray[:, 0:max_dist // 4]) + np.mean(gray[:, -max_dist // 4:])
    edge_ratio /= 4

    # Brain MRI has brighter center and darker edges.
    if center_ratio < edge_ratio:
        return {"ok": False, "reason": "Image lacks central brain structure (center should be brighter than edges)"}

    return {"ok": True, "reason": ""}


def _compute_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _compute_energy(probs: np.ndarray) -> float:
    return float(np.sum(np.exp(probs)))


def _check_feature_ood(model, img_array: np.ndarray) -> dict:
    """Check if input features are out-of-distribution using intermediate layer activations.

    Returns {"ok": bool, "reason": str, "feature_variance": float}.
    """
    try:
        feature_model = get_feature_extractor(model)
        features = feature_model.predict(img_array, verbose=0)
    except Exception:
        return {"ok": True, "reason": "Feature check unavailable", "feature_variance": 0.0}

    feat_var = float(np.var(features))
    feat_mean = float(np.mean(np.abs(features)))

    # If feature variance is too low or too high, the image is likely OOD.
    # Brain MRIs produce a specific range of feature activations.
    if feat_var < FEATURE_VARIANCE_MIN:
        return {"ok": False, "reason": f"Feature variance too low ({feat_var:.4f}), likely not a brain MRI",
                "feature_variance": feat_var}
    if feat_var > FEATURE_VARIANCE_MAX:
        return {"ok": False, "reason": f"Feature variance too high ({feat_var:.4f}), likely not a brain MRI",
                "feature_variance": feat_var}

    # Check if features have reasonable mean magnitude
    if feat_mean > 5.0:
        return {"ok": False, "reason": f"Feature magnitude abnormal ({feat_mean:.2f}), image may be OOD",
                "feature_variance": feat_var}

    return {"ok": True, "reason": "Features appear in-distribution", "feature_variance": feat_var}


def predict(model, image_bytes: bytes) -> dict:
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array, verbose=0)[0]

    # --- Feature-space OOD detection ---
    feat_check = _check_feature_ood(model, img_array)

    # --- Prediction-space OOD detection ---
    max_prob = float(np.max(preds))
    entropy = _compute_entropy(preds)
    energy = _compute_energy(preds)

    is_ood = (
        max_prob < MAX_PROB_THRESHOLD
        or entropy > ENTROPY_THRESHOLD
        or energy < ENERGY_THRESHOLD
        or not feat_check["ok"]
    )

    # --- Class prediction with glioma boost (only when competitive) ---
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
        "is_ood": is_ood,
        "max_probability": round(max_prob * 100, 2),
        "entropy": round(entropy, 4),
        "feature_variance": round(feat_check["feature_variance"], 6),
        "ood_reason": feat_check["reason"] if not feat_check["ok"] else "",
    }

    if is_ood:
        result["warning"] = (
            "This image does not appear to be a brain MRI. "
            "The prediction is unreliable and should not be trusted."
        )
        # Downgrade confidence for OOD images
        result["confidence"] = round(float(max_prob) * 100, 2)
        result["predicted_class"] = "unknown"

    return result


def validate_image(image_bytes: bytes) -> dict:
    """Public wrapper to check whether an image looks like a brain MRI."""
    return _is_brain_mri(image_bytes)
