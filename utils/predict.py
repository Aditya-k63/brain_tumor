import numpy as np
from keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf
import io
import os


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLIOMA_THRESHOLD = 0.10


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


def predict(model, image_bytes: bytes) -> dict:
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array, verbose=0)[0]

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
