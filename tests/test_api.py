import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient


def create_test_image(width=224, height=224, format="PNG"):
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def create_test_image_bytes(width=224, height=224):
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestPreprocessImage:
    def test_returns_correct_shape(self):
        from utils.predict import preprocess_image
        img_bytes = create_test_image_bytes()
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 224, 224, 3)

    def test_returns_float32(self):
        from utils.predict import preprocess_image
        img_bytes = create_test_image_bytes()
        result = preprocess_image(img_bytes)
        assert result.dtype == np.float32

    def test_handles_jpeg(self):
        from utils.predict import preprocess_image
        img_bytes = create_test_image(format="JPEG")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 224, 224, 3)

    def test_handles_different_sizes(self):
        from utils.predict import preprocess_image
        for size in [(100, 100), (512, 512), (64, 256)]:
            img = Image.new("RGB", size, color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            result = preprocess_image(buf.read())
            assert result.shape == (1, 224, 224, 3)


class TestPredict:
    def test_returns_expected_keys(self):
        from utils.predict import predict
        model = self._get_mock_model()
        img_bytes = create_test_image_bytes()
        result = predict(model, img_bytes)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predicted_class_is_valid(self):
        from utils.predict import predict, CLASS_NAMES
        model = self._get_mock_model()
        img_bytes = create_test_image_bytes()
        result = predict(model, img_bytes)
        assert result["predicted_class"] in CLASS_NAMES

    def test_probabilities_sum_to_100(self):
        from utils.predict import predict
        model = self._get_mock_model()
        img_bytes = create_test_image_bytes()
        result = predict(model, img_bytes)
        total = sum(result["probabilities"].values())
        assert abs(total - 100.0) < 1.0

    def test_confidence_is_between_0_and_100(self):
        from utils.predict import predict
        model = self._get_mock_model()
        img_bytes = create_test_image_bytes()
        result = predict(model, img_bytes)
        assert 0 <= result["confidence"] <= 100

    def _get_mock_model(self):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


class TestGradCAM:
    def test_returns_png_bytes(self):
        from utils.gradcam import generate_gradcam
        model = self._get_mock_model()
        img_bytes = create_test_image_bytes()
        result = generate_gradcam(model, img_bytes)
        assert isinstance(result, bytes)
        assert len(result) > 0
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"

    def test_output_is_rgb_image(self):
        from utils.gradcam import generate_gradcam
        model = self._get_mock_model()
        img_bytes = create_test_image_bytes()
        result = generate_gradcam(model, img_bytes)
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def _get_mock_model(self):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


class TestAPIEndpoints:
    def setup_method(self):
        import main
        self.client = TestClient(main.app)

    def test_home(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "version" in response.json()

    def test_health_without_model(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_rejects_invalid_file_type(self):
        response = self.client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_rejects_empty_file(self):
        response = self.client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 400

    def test_predict_rejects_oversized_file(self):
        large_image = b"\x89PNG" + b"\x00" * (11 * 1024 * 1024)
        response = self.client.post(
            "/predict",
            files={"file": ("large.png", large_image, "image/png")}
        )
        assert response.status_code == 400

    def test_predict_rejects_corrupt_image(self):
        response = self.client.post(
            "/predict",
            files={"file": ("corrupt.png", b"not a real png", "image/png")}
        )
        assert response.status_code == 400

    def test_gradcam_rejects_invalid_file_type(self):
        response = self.client.post(
            "/predict/gradcam",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_batch_rejects_too_many_files(self):
        files = [("files", (f"img{i}.png", create_test_image_bytes(), "image/png")) for i in range(11)]
        response = self.client.post("/predict/batch", files=files)
        assert response.status_code == 400

    def test_batch_accepts_valid_files(self):
        files = [("files", (f"img{i}.png", create_test_image_bytes(), "image/png")) for i in range(3)]
        response = self.client.post("/predict/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3
