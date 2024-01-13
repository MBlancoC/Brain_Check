import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Cargar el modelo YOLOv8
MODEL_PATH = 'yolov8_W.pt'  # Asegúrate de que este sea el camino correcto al modelo
model = YOLO(MODEL_PATH)

st.title('Detector de Tumores con YOLOv8')

uploaded_file = st.file_uploader("Sube una imagen en formato jpg o png", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Preparar la imagen para la inferencia
    image_for_model = image.resize((224, 224))  # Cambio a 224x224

    # Realizar la predicción
    results = model(image_for_model)

    # Procesar los resultados
    tumor_present = any(i.probs.data[1].item() > 0.5 for i in results)
    
    if tumor_present:
        st.write("El modelo predice que la imagen SÍ contiene un tumor.")
    else:
        st.write("El modelo predice que la imagen NO contiene un tumor.")
