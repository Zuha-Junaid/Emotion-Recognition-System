import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="Facial Emotion AI", layout="wide")

# Function to load model safely
@st.cache_resource
def load_custom_model():
    model_path = 'emotion_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

st.title("🧠 Facial Emotion Recognition Dashboard")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["1. Dataset EDA", "2. Model Performance", "3. Emotion Predictor"])

# --- PAGE 1: EDA ---
if page == "1. Dataset EDA":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Emotion Distribution")
        if os.path.exists('distribution.png'):
            st.image('distribution.png')
        else:
            st.warning("Run eda.py first to generate this chart.")
            
    with col2:
        st.subheader("Dataset Samples")
        if os.path.exists('samples.png'):
            st.image('samples.png')
        else:
            st.warning("Run eda.py first to generate sample images.")

# --- PAGE 2: PERFORMANCE ---
elif page == "2. Model Performance":
    st.header("Model Evaluation Metrics")
    if os.path.exists('confusion_matrix.png'):
        st.image('confusion_matrix.png', caption="Confusion Matrix Heatmap")
    else:
        st.warning("Run evaluate_model.py after training to see performance.")

# --- PAGE 3: PREDICTOR ---
elif page == "3. Emotion Predictor":
    st.header("Predict Emotion from Image")
    uploaded_file = st.file_uploader("Choose a photo of a face...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Display the Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        
        # 2. Preprocess for Custom CNN (48x48 Grayscale)
        # We convert to 'L' (Grayscale) and resize to 48x48
        image_gray = image.convert('L').resize((48, 48))
        img_array = np.array(image_gray) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1)) # Add batch and channel dims
        
        # 3. Prediction Logic
        model = load_custom_model()
        
        if model is not None:
            try:
                prediction = model.predict(img_array)
                emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                result = emotions[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                st.divider()
                st.success(f"### The model detects: **{result}**")
                st.info(f"Confidence Level: **{confidence:.2f}%**")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("🚨 Model file 'emotion_model.h5' not found! Please ensure training is complete.")