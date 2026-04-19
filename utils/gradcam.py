import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2
import io

LAST_CONV_LAYER = "top_activation"

def make_gradcam_heatmap(img_array, model):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # only positive gradients
    pooled_grads = tf.reduce_mean(
        tf.maximum(grads, 0),
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_gradcam(model, image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model)

    orig = np.array(img_resized)
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    # smooth upscaling
    heatmap_resized = cv2.resize(
        heatmap, (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # gamma correction for contrast boost
    heatmap_resized = np.power(heatmap_resized, 0.7)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # stronger overlay
    superimposed = cv2.addWeighted(orig_bgr, 0.5, heatmap_colored, 0.5, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    result_img = Image.fromarray(superimposed_rgb)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()