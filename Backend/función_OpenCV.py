import Backend.parametros as p
from tensorflow.keras.models import load_model #ESTE IMPORT DEBE ELIMINARSE DE app.y
from tensorflow.keras.preprocessing.image import img_to_array #ESTE IMPORT DEBE ELIMINARSE DE app.y
import numpy as np
import streamlit as st
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image
def modelo_opencv(image):
    model = load_model(p.Path_CV)
    processed_image = preprocess_image(image, target_size=(32, 32))
    prediction = model.predict(processed_image)
    prediction = np.argmax(prediction, axis=1)
    result_text = (f"<p style='color:green;'> La imagen no tiene tumor.</p>"
                    if prediction == 0
                    else f"<p style='color:red;'> La imagen SÍ tiene tumor.</p>")
    st.markdown(result_text, unsafe_allow_html=True)
    return("La imagen no tiene un tumor." if prediction==0 else "La imagen SÍ tiene un tumor.")
#La función preprocess image es la misma de la app. Debe ser eliminada de app.py
