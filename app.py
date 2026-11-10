import streamlit as st
import cv2
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('modelS.pkl', 'rb'))

st.title("🍎 Fresh vs Rotten Fruit Classifier")
st.write("Upload an image of the fruit for freshness classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Uploaded Image")
    
    if st.button("Predict"):
        img_resized = cv2.resize(opencv_image, (64, 64))
        hist = cv2.calcHist([img_resized], [0, 1, 2], None, [4,4,4], [0, 256, 0, 256, 0, 256])
        features = cv2.normalize(hist, hist).flatten().reshape(1, -1)
        
        prediction = model.predict(features)
        st.success(f"Result: {"Fresh" if prediction[0] == 0 else "Rotten"}")
        st.info("This is the deployment step! Once the model is saved using pickle, the prediction will appear here immediately.")