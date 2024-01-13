from ultralytics import YOLO
import Backend.parametros as p
import streamlit as st

def modelo_yolo(image):

    model = YOLO(p.Path_YL)

    image_for_model = image.resize((224, 224))  # Cambio a 224x224

    # Realizar la predicción
    results = model(image_for_model)

    # Procesar los resultados
    tumor_present = any(i.probs.data[1].item() > 0.5 for i in results)

    if tumor_present:
        st.write("El modelo predice que la imagen SÍ contiene un tumor.")
    else:
        st.write("El modelo predice que la imagen NO contiene un tumor.")
