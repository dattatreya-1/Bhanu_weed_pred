import streamlit as st
import numpy as np
import cv2
import pickle
import tensorflow as tf
from PIL import Image
import io

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Broadleaf Weed Prediction",
    page_icon="🌿",
    layout="centered"
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="leaf_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- LOAD LABEL ENCODER ----------
@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

le = load_encoder()

# ---------- UI ----------
st.title("🌿 Broadleaf Weed Prediction")
st.write("Upload a leaf image to predict weed class")

uploaded = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# ---------- PREDICTION ----------
if uploaded is not None:

    try:
        # read file safely
        file_bytes = uploaded.getvalue()

        image = Image.open(io.BytesIO(file_bytes))
        img = np.array(image)

        # grayscale fix
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # alpha channel fix
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # convert RGB → BGR (important if trained using cv2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.image(img, caption="Uploaded Image")

        # preprocessing
        img_resized = cv2.resize(img, (64, 64))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)

        with st.spinner("Analyzing leaf..."):
            interpreter.set_tensor(input_details[0]['index'], img_resized)
            interpreter.invoke()

            pred = interpreter.get_tensor(output_details[0]['index'])
            class_index = np.argmax(pred)
            confidence = np.max(pred)

            class_name = le.inverse_transform([class_index])

        st.success(f"Prediction : {class_name[0]}")
        st.info(f"Confidence : {confidence*100:.2f}%")

    except Exception as e:
        st.error("Error processing image. Please upload a valid leaf image.")
        st.write(e)
