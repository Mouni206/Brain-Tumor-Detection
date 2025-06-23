import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from util import classify


model_path = r'Your model path'
try:
    model = load_model(model_path)
    st.success(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


st.title('🧠 Brain Tumor Classification')
st.header('Please upload a brain MRI image')


file = st.file_uploader('Upload MRI Image', type=['jpeg', 'jpg', 'png'])

if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        class_name, conf_score = classify(image, model, class_names)

        st.markdown(f"## 🧠 Prediction: **{class_name.upper()}**")
        st.markdown(f"### 🔍 Confidence Score: **{conf_score * 100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
