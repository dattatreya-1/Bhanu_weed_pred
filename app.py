import streamlit as st
import numpy as np
import cv2
import pickle
import tflite_runtime.interpreter as tflite

# load tflite model
interpreter = tflite.Interpreter(model_path="leaf_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("🌿 Broadleaf Weed Prediction")

uploaded = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","png","jpeg","tif","tiff"]
)

if file is not None:

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(pred)

    class_name = le.inverse_transform([class_index])

    st.success(f"Prediction : {class_name[0]}")
