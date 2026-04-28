import streamlit as st
import requests
from PIL import Image
import io
import os

API_URL = os.environ.get("API_URL", "https://brain-tumor-latest-c2vy.onrender.com")


st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="wide"
)

st.title("Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI scan to classify tumor type and visualize GradCAM.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** EfficientNetB0  
    **Classes:**
    - 🔴 Glioma
    - 🟠 Meningioma
    - 🟢 No Tumor
    - 🔵 Pituitary
    
    **Glioma threshold:** 0.30  
    *(tuned for higher recall in medical context)*
    """)
    st.divider()
    st.caption("Built with FastAPI + TensorFlow + Streamlit")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan in JPG or PNG format"
)

if uploaded_file:
    col1, col2, col3 = st.columns(3)

    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes))

    with col1:
        st.subheader("Original MRI")
        st.image(img, use_container_width=True)

    with st.spinner("Analyzing MRI..."):
        try:
            pred_response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
            )
            pred_response.raise_for_status()
            result = pred_response.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    with st.spinner("Generating GradCAM..."):
        try:
            gradcam_response = requests.post(
                f"{API_URL}/predict/gradcam",
                files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
            )
            gradcam_response.raise_for_status()
            gradcam_img = Image.open(io.BytesIO(gradcam_response.content))

        except Exception as e:
            gradcam_img = None
            st.warning(f"GradCAM failed: {e}")

    with col2:
        st.subheader("Prediction")

        predicted = result["predicted_class"]
        confidence = result["confidence"]

        color_map = {
            "glioma": "🔴",
            "meningioma": "🟠",
            "notumor": "🟢",
            "pituitary": "🔵"
        }

        st.markdown(f"### {color_map[predicted]} {predicted.upper()}")
        st.metric("Confidence", f"{confidence}%")

        if predicted != "notumor":
            st.warning(" Tumor detected. Please consult a medical professional.")
        else:
            st.success("No tumor detected.")

        st.divider()
        st.subheader("All Probabilities")

        probs = result["probabilities"]
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
            st.progress(
                prob / 100,
                text=f"{color_map[cls]} {cls}: {prob}%"
            )

    with col3:
        st.subheader("GradCAM Overlay")
        if gradcam_img:
            st.image(gradcam_img, use_container_width=True)
        else:
            st.info("GradCAM not available")

    st.divider()
