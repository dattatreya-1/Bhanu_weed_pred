import streamlit as st
import numpy as np
import cv2
import pickle
import tensorflow as tf

# ---------- Load TFLite Model ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="leaf_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- Load Label Encoder ----------
@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

le = load_encoder()

# ---------- UI ----------
st.title("🌿 Broadleaf Weed Prediction")

uploaded = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "png", "jpeg", "tif", "tiff"]
)

if uploaded is not None:

   from PIL import Image

image = Image.open(uploaded)
img = np.array(image)

# if grayscale convert to RGB
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# if RGBA convert to RGB
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------- Preprocessing ----------
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # ---------- Prediction ----------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(pred)

    class_name = le.inverse_transform([class_index])

    st.success(f"Prediction : {class_name[0]}")
