import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model("broadleaf_cnn_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("🌿 Broadleaf Classification System")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    class_name = le.inverse_transform([class_index])

    st.success(f"Predicted Class : {class_name[0]}")
