import io
import streamlit as st
from PIL import Image
import base64
from Backend.predict_api import predict_image_classification_sample
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import Backend.parametros as p
from Backend.vision_api import analyze_image_with_gpt4

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

model1 = load_model(p.Path_CV) #path

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    if model_choice == 'OpenCV':
        # llamar a la funcion de predicción de OpenCV
        processed_image = preprocess_image(image, target_size=(32, 32))
        prediction = model1.predict(processed_image)
        prediction = np.argmax(prediction, axis=1)
        if prediction == 0:
            st.write("The MRI image is classified as: No Tumor")
        else:
            st.write("The MRI image is classified as: Yes Tumor")

    if model_choice == 'Vertex':
        # llamar a la funcion de predicción de Vertex AI y capturar la respuesta


        # Guardar la imagen en un archivo temporal para Vertex AI
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Llamar a la función de predicción de Vertex AI y capturar la respuesta
        prediction_response = predict_image_classification_sample(
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
    if model_choice == 'YoloV8':
        # Llamar a la función de predicción de YoloV8 y capturar la respuesta
        pass

    # Convertir la imagen a base64 para la API de OpenAI
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Caja de texto para obtener la pregunta del usuario para GPT-4
    user_question = st.text_input("Escribe tu pregunta sobre la imagen para GPT-4:")

    # Botón de envío
    if st.button('Enviar pregunta'):
        try:
            # Usar la función analyze_image_with_gpt4 para realizar la petición a OpenAI
            response = analyze_image_with_gpt4(image_data, user_question)

            # Mostrar la respuesta
            st.write("Respuesta de GPT-4 Vision:")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error("Error al procesar la solicitud: " + str(e))
