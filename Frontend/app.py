import io
import streamlit as st
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import openai
import Backend.parametros as p
from Backend.funcion_yolo import modelo_yolo
from Backend.vision_api import upload_multiple_files
from Backend.Vertex_AI import predict_vertex

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

st.title("Brain Check")
model_choice = st.selectbox('Elige el modelo:', ('OpenCV', 'Vertex', 'YoloV8'))

uploaded_files = st.file_uploader("Carga tus imágenes aquí", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen cargada', use_column_width=True)

            if model_choice == 'OpenCV':
                model = load_model(p.Path_CV)
                processed_image = preprocess_image(image, target_size=(32, 32))
                prediction = model.predict(processed_image)
                prediction = np.argmax(prediction, axis=1)
                if prediction == 0:
                    st.write("La imagen no tiene tumor.")
                else:
                    st.write("La imagen tiene tumor.")

            elif model_choice == 'Vertex':
                prediction_response = predict_vertex(uploaded_file)  # Asegúrate que esta función maneje un solo archivo
                if prediction_response:
                    for prediction in prediction_response:
                        if 'displayNames' in prediction and 'confidences' in prediction and 'ids' in prediction:
                            for displayName, id, confidence in zip(prediction['displayNames'], prediction['ids'], prediction['confidences']):
                                st.write(f"Resultado de Vertex AI: {displayName}, Confianza: {confidence:.2f}")

            elif model_choice == 'YoloV8':
                modelo_yolo(image)

            elif model_choice == "VGG-16":
                pass

            elif model_choice == "RestNet":
                pass

if uploaded_files:
    user_question = st.text_input("Escribe tu pregunta sobre la imagen para GPT-4:")

    if user_question and st.button('Enviar pregunta'):
        try:
            response = upload_multiple_files(uploaded_files, user_question)
            if response:
                st.write("Respuesta de GPT-4 Vision:")
                st.write(response.choices[0].message.content)
            else:
                st.error("Por favor, sube al menos una imagen y escribe una pregunta.")
        except Exception as e:
            st.error("Error al procesar la solicitud: " + str(e))
