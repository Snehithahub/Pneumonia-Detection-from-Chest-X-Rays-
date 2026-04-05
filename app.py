import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

model = load_model('pneumonia_cnn.h5')

def predict(img):
    img_resized = img.resize((224,224))
    x = np.array(img_resized)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    return "Pneumonia" if pred > 0.5 else "Normal", pred

st.title("Pneumonia Detection from Chest X-Rays")

uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded X-Ray", use_column_width=True)
    label, confidence = predict(img)
    st.write(f"Prediction: {label} ({confidence:.2f})")

    # Grad-CAM
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    from gradcam import get_gradcam  # save Grad-CAM function in gradcam.py
    heatmap_img = get_gradcam(uploaded_file, model)
    st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), caption="Grad-CAM Heatmap", use_column_width=True)
