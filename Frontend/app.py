import os
import io
import streamlit as st
from PIL import Image
import base64
from Backend import predict_api
from Backend.predict_api import predict_image_classification_sample
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import openai

openai.api_key = "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"

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image



st.title("Brain Check")

model_choice = st.selectbox('Elige el modelo:', ('OpenCV', 'Vertex'))

# Subida de archivo para ambas APIs
uploaded_file = st.file_uploader("Carga tu imagen aquí", type=["jpg", "png"])

# Path_CV = "Modelos/OpenCV/modelcv.h5"
model1 = load_model('C:/Users/marsi/PycharmProjects/chatgpt_api/mi_modelo_OpenCV.h5') #path

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    if model_choice == 'OpenCV':
        processed_image = preprocess_image(image, target_size=(32, 32))
        prediction = model1.predict(processed_image)
        prediction = np.argmax(prediction, axis=1)
        if prediction == 0:
            st.write("The MRI image is classified as: No Tumor")
        else:
            st.write("The MRI image is classified as: Yes Tumor")

    if model_choice == 'Vertex':
        # Guardar la imagen en un archivo temporal para Vertex AI
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Llamar a la función de predicción de Vertex AI y capturar la respuesta
        prediction_response = predict_api.predict_image_classification_sample(
            project="263184688391",
            endpoint_id="2305326238748639232",
            location="us-central1",
            filename="temp_image.jpg"
        )

        # Mostrar los resultados de Vertex AI en la aplicación
        if prediction_response:
            for prediction in prediction_response:
                if 'displayNames' in prediction and 'confidences' in prediction and 'ids' in prediction:
                    for displayName, id, confidence in zip(prediction['displayNames'], prediction['ids'], prediction['confidences']):
                        st.write(f"Resultado de Vertex AI: {displayName}, Confianza: {confidence:.2f}")

    # Convertir la imagen a base64 para la API de OpenAI
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Caja de texto para obtener la pregunta del usuario para GPT-4
    user_question = st.text_input("Escribe tu pregunta sobre la imagen para GPT-4:")

    # Agregar system, con el rol, contexto, segmentacion del cerebro y ubicacion
    # del tumor (codigo previo para encontrar ubicacion)

    # Botón de envío
    if st.button('Enviar pregunta'):
        try:
            # Realizar la petición a OpenAI
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_question},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
                        ],
                    }
                ],
                max_tokens=300,
            )

            # Mostrar la respuesta
            st.write("Respuesta de GPT-4 Vision:")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error("Error al procesar la solicitud: " + str(e))
