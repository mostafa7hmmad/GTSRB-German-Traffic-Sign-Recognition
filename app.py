import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import pickle
import os

st.set_page_config(page_title="GTSRB Recognition", layout="centered")
st.title("German Traffic Sign Recognition")

# Paths (files should be in the same directory as this script)
MODEL_PATH = "model_trained.h5"
CLASSES_PATH = "class_indices.pkl"
# Possible locked ODS filenames
ODS_FILES = [
    ".~lock.ClassesInformation.ods#",
    "_lock.ClassesInformationStrong.ods#"
]

# Load model from local file system
def load_local_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load class mapping from ODS or pickle
def load_class_mapping():
    # 1. Try each locked ODS spreadsheet
    for ods in ODS_FILES:
        if os.path.exists(ods):
            try:
                df = pd.read_excel(ods, engine='odf')
                if {'index', 'label'}.issubset(df.columns):
                    mapping = dict(zip(df['index'], df['label']))
                    st.success(f"Class mapping loaded from {ods}")
                    return mapping
                else:
                    st.warning(f"Spreadsheet {ods} missing 'index'/'label' columns.")
            except Exception as e:
                st.error(f"Failed to read spreadsheet {ods}: {e}")
    # 2. Try pickle file
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'rb') as f:
                mapping = pickle.load(f)
            st.success(f"Class mapping loaded from {CLASSES_PATH}")
            return mapping
        except Exception as e:
            st.error(f"Failed to load pickle mapping: {e}")
    # 3. Fallback: numeric labels
    st.warning("No class mapping found; using numeric class indices.")
    return {}

# Initialize
model = load_local_model(MODEL_PATH)
index_to_label = load_class_mapping()

# Upload image for prediction
download_image = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Ensure 'model_trained.h5' is in the script directory.")
    elif not index_to_label:
        st.error("Class mapping not loaded. Ensure one of the ODS lock files or 'class_indices.pkl' are present.")
    elif download_image is None:
        st.error("Please upload an image first.")
    else:
        try:
            img = Image.open(download_image).convert('RGB').resize((32, 32), Image.LANCZOS)
            arr = np.expand_dims(np.array(img) / 255.0, 0)
            preds = model.predict(arr)
            idx = int(np.argmax(preds))
            conf = float(preds[0][idx])

            st.image(img, caption="Input Image", use_column_width=True)
            label = index_to_label.get(idx, f"Class {idx}")
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {conf*100:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("_Ensure 'model_trained.h5' and at least one of the ODS lock files or 'class_indices.pkl' are present in the script directory before running._")
