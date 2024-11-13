from tensorflow.keras.models import load_model

model = load_model('skin_cancer_model.h5')  # Replace with your model's filename

import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2  # Import OpenCV

# Load your trained model (as shown in the previous step)

st.title('Skin Cancer Detection')

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Preprocess the image
    img = cv2.resize(img, (224, 224))  # Resize to match your model input
    img = img / 255.  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)

    # Assuming your model outputs probabilities for two classes (benign, malignant)
    if prediction[0][0] > 0.5:
        result = "Malignant"
        confidence = prediction[0][0]
    else:
        result = "Benign"
        confidence = 1 - prediction[0][0]

    # Display the results
    st.image(img[0], caption="Uploaded Image", use_container_width=True)
    st.write(f"Prediction: **{result}**")
    st.write(f"Confidence: **{confidence:.2f}**")