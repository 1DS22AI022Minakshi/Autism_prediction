import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("autism_cnn_model.h5")

# Get expected input size from model
img_height, img_width, channels = model.input_shape[1:]

# Streamlit UI
st.title("Autism Prediction Using Deep Learning")
st.write("Upload a child's image to predict whether they are autistic or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image to model's expected input
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize (important!)

    # Predict
    prediction = model.predict(img_array)

    # Display prediction (assuming binary classification with sigmoid)
    if prediction[0][0] > 0.5:
        st.error(f"Prediction:Non-Autistic (Confidence: {prediction[0][0] * 100:.2f}%)")
    else:
        st.success(f"Prediction: Autistic (Confidence: {(1 - prediction[0][0]) * 100:.2f}%)")
