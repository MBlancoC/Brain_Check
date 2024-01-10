import os
import io
import streamlit as st
from PIL import Image
import base64
from Backend.predict_api import predict_image_classification_sample
from Backend.vision_api import analyze_image_with_gpt4

st.title("Brain Check")

# Subida de archivo para ambas APIs
uploaded_file = st.file_uploader("Carga tu imagen aquí", type=["jpg", "png"])
if uploaded_file is not None:
    # Mostrar la imagen cargadaåç
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

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

    # Inicializar el estado de la sesión para el chat
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Mostrar mensajes anteriores
    for message in st.session_state["messages"]:
        role, content = message["role"], message["content"]
        with st.chat_message(role):
            st.markdown(content)

    # Input de chat
    user_input = st.chat_input("Chat with GPT-4 Vision:")

    # Procesar la entrada del usuario
    if user_input:
        # Convertir la imagen a base64 para GPT-4
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Análisis de la imagen con GPT-4
        try:
            response = analyze_image_with_gpt4(image_data, user_input)
            assistant_reply = response.choices[0].message.content

            # Añadir pregunta y respuesta al historial
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
        except Exception as e:
            st.error("Error al procesar la imagen: " + str(e))
            print(e)


#revisar tema memoria para almacenar las preguntas y respuestas
