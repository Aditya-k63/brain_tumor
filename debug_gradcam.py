# run this separately in a new python file - debug_gradcam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model/brain_tumor_final.keras')

# load a glioma image from your test set
img_path = r"C:\Users\adity\Desktop\braintumor\test_glioma.jpg"  # change path

img = Image.open(img_path).convert("RGB").resize((224, 224))
img_array = np.array(img, dtype=np.float32)
img_array_processed = preprocess_input(img_array.copy())
img_array_processed = np.expand_dims(img_array_processed, axis=0)

# check prediction first
preds = model.predict(img_array_processed)[0]
print("Predictions:", {n: round(float(p)*100, 2) for n, p in zip(["glioma","meningioma","notumor","pituitary"], preds)})

# gradcam
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer("top_activation").output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array_processed)
    pred_index = tf.argmax(predictions[0])
    class_channel = predictions[:, pred_index]

grads = tape.gradient(class_channel, conv_outputs)
print("Grads max:", tf.reduce_max(grads).numpy())
print("Grads min:", tf.reduce_min(grads).numpy())
print("Conv output shape:", conv_outputs.shape)

pooled_grads = tf.reduce_mean(tf.maximum(grads, 0), axis=(0, 1, 2))
print("Pooled grads max:", tf.reduce_max(pooled_grads).numpy())
print("Non-zero pooled grads:", tf.reduce_sum(tf.cast(pooled_grads > 0, tf.float32)).numpy())

conv_out = conv_outputs[0]
heatmap = conv_out @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0)
heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
heatmap = heatmap.numpy()

print("Heatmap shape:", heatmap.shape)
print("Heatmap max:", heatmap.max())
print("Heatmap min:", heatmap.min())
print("Heatmap mean:", heatmap.mean())

# plot raw heatmap
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.title("Raw Heatmap")
plt.axis("off")

# overlay
heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
heatmap_resized = np.power(heatmap_resized, 0.7)
orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(orig_bgr, 0.5, heatmap_colored, 0.5, 0)
superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

plt.subplot(1, 3, 3)
plt.imshow(superimposed_rgb)
plt.title("GradCAM Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_debug.png", dpi=150)
plt.show()
print("Saved to gradcam_debug.png")