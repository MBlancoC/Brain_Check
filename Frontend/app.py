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
    # Establecer el número de columnas por fila en el grid
    cols_per_row = 5
    # Crear contenedores para las filas
    rows = [st.container() for _ in range((len(uploaded_files) + cols_per_row - 1) // cols_per_row)]
    # Crear las columnas dentro de cada contenedor de fila
    cols_in_row = [row.columns(cols_per_row) for row in rows]

    # Para almacenar las predicciones y asociarlas con sus imágenes
    predictions = []

    # Procesar cada archivo subido y realizar la predicción
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col = cols_in_row[i // cols_per_row][i % cols_per_row]

            with col:
                # Mostrar la imagen
                st.image(image, use_column_width=True)
                file_name = uploaded_file.name

                # Procesamiento específico de cada modelo
                if model_choice == 'OpenCV':
                    model = load_model(p.Path_CV)
                    processed_image = preprocess_image(image, target_size=(32, 32))
                    prediction = model.predict(processed_image)
                    prediction = np.argmax(prediction, axis=1)
                    result_text = f"{file_name}: La imagen no tiene tumor." if prediction == 0 else f"{file_name}: La imagen tiene tumor."

                elif model_choice == 'Vertex':
                    prediction_response = predict_vertex(uploaded_file)  # Asegúrate de que esta función acepte un archivo y devuelva una predicción
                    # Asume que predict_vertex devuelve un diccionario con la predicción y la confianza
                    if prediction_response['prediction'] == 'No Tumor':
                        result_text = f"Vertex AI - {file_name}: La imagen no tiene tumor. Confianza: {prediction_response['confidence']:.2f}"
                    else:
                        result_text = f"Vertex AI- {file_name}: La imagen tiene tumor. Confianza: {prediction_response['confidence']:.2f}"

                elif model_choice == 'YoloV8':
                    result_text = modelo_yolo(image)  # Asume que modelo_yolo devuelve un string con el resultado

                elif model_choice == "VGG-16":
                    pass

                elif model_choice == "RestNet":
                    pass
                predictions.append(result_text)
                print(predictions)
            # Mostrar las predicciones debajo de cada imagen

    for i, prediction_text in enumerate(predictions):
        col = cols_in_row[i // cols_per_row][i % cols_per_row]
        with col:
            st.caption(prediction_text)

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
