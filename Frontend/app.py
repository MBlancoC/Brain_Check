import io
import streamlit as st
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import openai
import Backend.parametros as p
from Backend.vision_api import analyze_image_with_gpt4
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
model_choice = st.selectbox('Elige el modelo:', ('OpenCV', 'Vertex', 'GPT-4', 'YoloV8'))

# Inicialización de la variable image
image = None

if model_choice in ['OpenCV', 'Vertex', 'YoloV8']:
    uploaded_file = st.file_uploader("Carga tu imagen aquí", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen cargada', use_column_width=True)

    if model_choice == 'OpenCV' and image is not None:
        model1 = load_model(p.Path_CV)
        processed_image = preprocess_image(image, target_size=(32, 32))
        prediction = model1.predict(processed_image)
        prediction = np.argmax(prediction, axis=1)
        if prediction == 0:
            st.write("The MRI image is classified as: No Tumor")
        else:
            st.write("The MRI image is classified as: Yes Tumor")

    if model_choice == 'Vertex' and image is not None:
        prediction_response = predict_vertex(uploaded_file)
        if prediction_response:
            for prediction in prediction_response:
                if 'displayNames' in prediction and 'confidences' in prediction and 'ids' in prediction:
                    for displayName, id, confidence in zip(prediction['displayNames'], prediction['ids'], prediction['confidences']):
                        st.write(f"Resultado de Vertex AI: {displayName}, Confianza: {confidence:.2f}")

    if model_choice == 'YoloV8' and image is not None:
        modelo_yolo(image)

    if image is not None:
        # Convertir la imagen a base64 para la API de OpenAI
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Caja de texto para obtener la pregunta del usuario para GPT-4
        user_question = st.text_input("Escribe tu pregunta sobre la imagen para GPT-4:")

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
                st.write("Respuesta de GPT-4 Vision:")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error("Error al procesar la solicitud: " + str(e))


if model_choice == 'GPT-4':
    uploaded_files = st.file_uploader("Chat with GPT-4 Vision", type=["jpg", "png"], accept_multiple_files=True)
    user_question = st.text_input("Escribe tu pregunta sobre la imagen para GPT-4:")

    if uploaded_files and user_question and st.button('Enviar pregunta a GPT-4'):
        try:
            response = upload_multiple_files(uploaded_files, user_question)
            if response:
                st.write("Respuesta de GPT-4 Vision:")
                st.write(response.choices[0].message.content)
            else:
                st.error("Por favor, sube al menos una imagen y escribe una pregunta.")
        except Exception as e:
            st.error("Error al procesar la solicitud: " + str(e))
