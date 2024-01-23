import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Carga el modelo VGG
model_vgg = load_model('Primer_VGG_model_6FN.keras')
st.title('MRI Tumor Classifier')
st.write('Please upload an MRI image for tumor classification.')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    target_size = (224, 224)
    processed_image = preprocess_image(image, target_size=target_size)
    prediction = model_vgg.predict(processed_image)
    prediction = np.argmax(prediction, axis=1)
    if prediction == 0:
        st.write("The MRI image is classified as: No Tumor")
    else:
        st.write("The MRI image is classified as: Yes Tumor")
