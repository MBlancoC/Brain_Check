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
    # Construir el mensaje
    mensaje = (f"<p style='color:green;'>La imagen no tiene un tumor.</p>"
               if not tumor_present
               else f"<p style='color:red;'>La imagen SÍ tiene un tumor.</p>")
    # Escribir el mensaje en la salida de streamlit
    st.markdown(mensaje, unsafe_allow_html=True)
    return("La imagen SÍ tiene un tumor." if tumor_present else "La imagen no tiene un tumor.")
