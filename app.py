import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("🌿 Plant Disease Detection App")

model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = ['Pepper_Bacterial_spot', 'Pepper_healthy', 'Potato_Early_blight', 'Potato_healthy', 'Tomato_Bacterial_spot', 'Tomato_healthy']

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    st.success(f"🌱 Predicted Disease: {result}")
