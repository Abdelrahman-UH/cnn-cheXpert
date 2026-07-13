"""
Streamlit Application for CheXpert Medical Image Classification.

Provides a premium, responsive medical dashboard UI to perform chest X-ray
classification using a trained EfficientNetB0 model.
"""

import os
import logging
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Set page config at the very beginning of Streamlit app
st.set_page_config(
    page_title="CheXpert Image Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils import (
    build_model,
    load_model_weights,
    preprocess_image,
    run_inference,
    CHEXPERT_LABELS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CheXpertApp")

# Path to the weights file
DEFAULT_WEIGHTS_PATH = "feature_extraction.weights.h5"


@st.cache_resource
def load_cached_model(weights_path: str):
    """
    Builds the model and loads weights. Cached to avoid reloading on each run.
    """
    logger.info(f"Caching: Building and loading model from {weights_path}")
    try:
        model = build_model()
        model = load_model_weights(model, weights_path)
        return model, None
    except Exception as e:
        logger.error(f"Error loading model in st.cache_resource: {e}")
        return None, str(e)


def create_dummy_chest_xray() -> Image.Image:
    """
    Generates a synthetic chest X-ray image for testing purposes.
    """
    # Create grayscale base image
    img = Image.new("L", (300, 300), color=15)
    draw = ImageDraw.Draw(img)

    # Draw a mock rib cage outline (simple geometric shapes)
    draw.ellipse([50, 40, 250, 280], fill=30)
    # Left and right lungs (darker areas)
    draw.ellipse([70, 60, 140, 240], fill=10)
    draw.ellipse([160, 60, 230, 240], fill=10)

    # Spine/mediastinum (brighter vertical strip)
    draw.rectangle([141, 40, 159, 280], fill=50)

    # Heart shape (middle-bottom left bulge)
    draw.ellipse([120, 150, 190, 230], fill=65)

    # Diaphragm domes
    draw.ellipse([50, 230, 145, 320], fill=40)
    draw.ellipse([155, 230, 250, 320], fill=40)

    # Add Gaussian-like noise to look like actual X-ray film grain
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Convert back to image
    xray_img = Image.fromarray(arr).convert("RGB")
    return xray_img


# Custom Styling using HTML & CSS
st.markdown("""
<style>
    /* Main Background & Accent styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Style st.button default styles */
    .stButton>button {
        background-color: #0284c7 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0369a1 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Dashboard Config</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

weights_file = st.sidebar.text_input("Model Weights Path", DEFAULT_WEIGHTS_PATH)
confidence_threshold = st.sidebar.slider(
    "Critical Finding Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help="Pathologies with prediction probability above this value will be highlighted as critical."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Clinical Project Details")
st.sidebar.markdown("""
- **Model**: EfficientNetB0 (CNN)
- **Dataset**: CheXpert (14 Labels)
- **Target Size**: 224x224x3
- **Environment**: Python 3.11.9
- **Libraries**: TensorFlow, Pillow, Pandas, Streamlit
""")

# --- Main Dashboard ---
st.markdown("# 🩺 CheXpert Chest X-Ray Diagnosis Dashboard")
st.markdown(
    "Deploying state-of-the-art Deep Learning models to assist clinicians with "
    "automated multi-label detection of thoracic pathologies from chest radiographs."
)
st.markdown("---")

# Verify weights file existence
if not os.path.exists(weights_file):
    st.error(
        f"⚠️ **Model weights file not found** at path: `{weights_file}`. "
        "Please ensure the file is present in the application directory."
    )
    st.stop()

# Initialize model
with st.spinner("🔄 Initializing neural network and loading weights..."):
    model, load_error = load_cached_model(weights_file)

if load_error:
    st.error(f"❌ **Failed to load model**: {load_error}")
    st.stop()

# Layout split: Input Panel vs Result Panel
col1, col2 = st.columns([1, 1])

uploaded_image = None

with col1:
    st.subheader("1. Radiograph Input")

    tab_upload, tab_sample = st.tabs(["📤 Upload Custom Image", "🖼️ Try a Sample Image"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a Chest X-ray image...",
            type=["png", "jpg", "jpeg"],
            help="Supported file types: PNG, JPG, JPEG"
        )
        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error opening uploaded image file: {e}")
                logger.error(f"File load error: {e}")

    with tab_sample:
        st.write("No image on hand? Generate and test a synthetic chest X-ray instantly:")
        if st.button("Generate Synthetic X-Ray"):
            uploaded_image = create_dummy_chest_xray()
            st.success("Generated synthetic chest X-ray!")

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Selected Chest X-Ray Input",
            use_container_width=True
        )

with col2:
    st.subheader("2. Diagnostic Results")

    if uploaded_image is None:
        st.info("ℹ️ Upload or generate a radiograph on the left to perform classification.")
    else:
        # Preprocessing
        with st.spinner("⏳ Preprocessing image (224x224x3)..."):
            try:
                processed_array = preprocess_image(uploaded_image)
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                logger.error(f"Preprocessing error: {e}")
                st.stop()

        # Prediction
        with st.spinner("🤖 Evaluating image with EfficientNetB0..."):
            try:
                predictions = run_inference(model, processed_array)
            except Exception as e:
                st.error(f"Error during model prediction: {e}")
                logger.error(f"Inference error: {e}")
                st.stop()

        # Display results
        st.success("Prediction complete!")

        # Sort predictions for list presentation
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Highlight Critical Findings
        critical_findings = [label for label, val in sorted_preds if val >= confidence_threshold]
        normal_findings = [label for label, val in sorted_preds if val < confidence_threshold]

        # Metric layout
        st.markdown("### 🚨 Critical Findings Summary")
        if critical_findings:
            st.markdown(
                f"The following findings met or exceeded the confidence threshold of "
                f"**{confidence_threshold * 100:.0f}%**:"
            )
            # Display findings as alert cards
            for finding in critical_findings:
                prob_val = predictions[finding]
                st.error(f"⚠️ **{finding}**: **{prob_val * 100:.1f}%** probability detected.")
        else:
            st.success(
                f"No findings exceeded the confidence threshold of "
                f"**{confidence_threshold * 100:.0f}%**."
            )

        st.markdown("---")
        st.markdown("### 📊 Pathologies Probability Distribution")

        # Create progress bars for all pathologies
        for label, val in sorted_preds:
            color_prefix = "🔴" if val >= confidence_threshold else "🟢"
            # Streamlit progress bar needs range [0.0, 1.0]
            st.write(f"{color_prefix} **{label}**: {val * 100:.1f}%")
            st.progress(val)

        # Optional: Plotly/Matplotlib visualization
        st.markdown("---")
        st.markdown("### 📉 Diagnosis Plot")
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = [x[0] for x in sorted_preds]
            values = [x[1] for x in sorted_preds]
            colors = ["#ef4444" if v >= confidence_threshold else "#10b981" for v in values]

            ax.barh(labels[::-1], values[::-1], color=colors[::-1])
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Probability")
            ax.set_title("Probability score per class")
            plt.tight_layout()

            st.pyplot(fig)
        except Exception as e:
            logger.error(f"Error plotting graph: {e}")
            st.warning("Could not generate probability plot.")
