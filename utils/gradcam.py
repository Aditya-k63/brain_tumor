import numpy as np
import tensorflow as tf
from keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2
import io


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        tf.maximum(grads, 0),
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)
    return heatmap.numpy()


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def generate_gradcam(model, image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array_preprocessed = preprocess_input(img_array.copy())
    img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)

    heatmap = make_gradcam_heatmap(img_array_preprocessed, model)

    orig = np.array(img_resized)
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(
        heatmap, (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap_resized = np.power(heatmap_resized, 0.7)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    superimposed = cv2.addWeighted(orig_bgr, 0.5, heatmap_colored, 0.5, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    result_img = Image.fromarray(superimposed_rgb)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()