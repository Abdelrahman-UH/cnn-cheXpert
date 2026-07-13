# CheXpert Chest X-Ray Diagnosis Dashboard

A production-ready Streamlit application that deploys a trained **EfficientNetB0** model to diagnose 14 thoracic pathologies from chest radiographs (X-rays).

---

## Technical Specifications
- **Python Version**: `3.11.9`
- **Core Technologies**: TensorFlow, Streamlit, Pandas, Pillow, Matplotlib
- **Base Model**: EfficientNetB0 (CNN)
- **Target Pathologies**: 14 distinct labels from the CheXpert dataset.

---

## Getting Started

Follow these steps to set up a clean, isolated environment and run the application.

### Step 1: Create a Virtual Environment
Create a virtual environment using Python `3.11.9` to avoid dependency conflicts with other global packages:

```bash
# Navigate to the workspace directory
cd "path/to/deployment data"

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
Install all required packages specified in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Model Weights Verification
The app requires the pre-trained weights file `feature_extraction.weights.h5` to load the trained parameters. 
Ensure `feature_extraction.weights.h5` is in the root of the project directory.

### Step 4: Run the Application
Start the Streamlit application:

```bash
streamlit run app.py
```

After running this command, your default web browser will open the dashboard at `http://localhost:8501`.

---

## Key Features

1. **Robust UI & UX**: Optimized dashboard styling with light/dark adaptive custom styling, intuitive column layouts, and high-quality charts.
2. **Flexible Image Input**: Users can upload chest X-ray images (PNG, JPG, JPEG) or click the **Generate Synthetic X-Ray** button to test the inference engine instantly with realistic simulated chest geometry.
3. **Clinical Priority View**: Pathologies are evaluated and categorized into *Critical Findings* based on an adjustable threshold (default 50%), allowing medical practitioners to immediately focus on high-probability issues.
4. **Performance Caching**: Uses `@st.cache_resource` to load the model structure and load model weights once, ensuring high responsiveness during multiple classification requests.
5. **Console Logging**: Complete step-by-step telemetry logs detailing preprocessing time, image mode changes, model initialization, and prediction latency.

---

## Troubleshooting & Debugging

- **Weights file error**: If you receive a warning about the model weights not found, double check that `feature_extraction.weights.h5` is placed in the active directory from which you ran `streamlit run app.py`.
- **Memory Errors**: TensorFlow might allocate a large amount of memory. Close background processes or run with CPU-only (default on Windows installations without custom WSL2 GPU setup).
- **Console Logs**: Watch the terminal output while using the app. You will see precise telemetry logs, for example:
  ```text
  2026-07-13 22:20:15 [INFO] CheXpertUtils: Instantiating EfficientNetB0 base model...
  2026-07-13 22:20:20 [INFO] CheXpertUtils: Loading weights from feature_extraction.weights.h5...
  2026-07-13 22:20:22 [INFO] CheXpertUtils: Weights loaded successfully in 2.132 seconds.
  2026-07-13 22:20:25 [INFO] CheXpertUtils: Preprocessing completed in 0.0042 seconds.
  2026-07-13 22:20:26 [INFO] CheXpertUtils: Inference run completed in 0.1250 seconds.
  ```
