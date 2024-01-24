import io
import streamlit as st
from PIL import Image
import base64
import numpy as np
import openai
import Backend.parametros as p
from Backend.funcion_yolo import modelo_yolo
from Backend.vision_api import upload_multiple_files
from Backend.Vertex_AI import predict_vertex
from streamlit_chat import message
from Backend.predict_api import predict_image_classification_sample
from Backend.vision_api import analyze_image_with_gpt4, get_chatgpt_response, update_chat
from Backend.función_OpenCV import modelo_opencv


# Asumiendo que logo_path es la ruta al archivo del logo
logo_path = p.PATH_LOGO

# Creación de las columnas para el logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(logo_path, width=200)

# Título de la aplicación
with col2:
    st.title("Brain Check")

# Definición de las precisiones de los modelos con estrellas
model_precisions = {
    'OpenCV': '★★★★☆ - 95%',
    'Vertex': '★★★★★ - 98.5%',
    'YoloV8': '★★★★★ - 97.9%',
    'VGG-16': '★★★☆☆ - 75%',  # Ejemplo, ajusta según sea necesario
    'RestNet': '★★★★☆ - 90%'  # Ejemplo, ajusta según sea necesario
}

# Creación de las opciones para el selectbox incluyendo las estrellas
model_options = ('OpenCV - ★★★★☆ - 95%', 'Vertex - ★★★★★ - 98.5%', 'YoloV8 - ★★★★★ - 97.9%')
model_choice = st.selectbox('Elige el modelo:', model_options)

uploaded_files = st.file_uploader("Carga tus imágenes aquí", type=["jpg", "png"], accept_multiple_files=True)
if uploaded_files:
    # Establecer el número de columnas por fila en el grid
    cols_per_row = 5
    # Crear contenedores para las filas
    rows = [st.container() for _ in range((len(uploaded_files) + cols_per_row - 1) // cols_per_row)]
    # Crear las columnas dentro de cada contenedor de fila
    cols_in_row = [row.columns(cols_per_row) for row in rows]
    # Para almacenar las predicciones y asociarlas con sus imágenes
    predictions = {}
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
                if model_choice == 'OpenCV - ★★★★☆ - 95%':
                    result_text = modelo_opencv(image)
                    predictions[file_name]=result_text
                elif model_choice == 'Vertex - ★★★★★ - 98.5%':
                    # Guardar la imagen en un archivo temporal para Vertex AI
                    with open("temp_image.jpg", "wb") as file:
                        file.write(uploaded_file.getbuffer())
                    # Llamar a la función de predicción de Vertex AI
                    prediction_response = predict_image_classification_sample(
                        project="309967433708",
                        endpoint_id="8459565498294599680",
                        location="us-central1",
                        filename="temp_image.jpg"
                    )
                    # Procesar la respuesta de la predicción
                    if prediction_response:
                        result_texts = []
                        for prediction in prediction_response:  # Itera directamente sobre la lista
                            displayNames = prediction.get('displayNames', [])
                            confidences = prediction.get('confidences', [])
                            for displayName, confidence in zip(displayNames, confidences):
                                color = "green" if displayName == "no" else "red"
                                if displayName == "no":
                                    result_texts.append(f"<p style='color:green;'> La imagen no tiene tumor.</p>")
                                else:
                                    result_texts.append(f"<p style='color:red;'> La imagen SÍ tiene tumor.</p>")

                        result_text = f"{file_name}: " + "; ".join(result_texts)
                        predictions[file_name]=result_text
                        st.markdown(result_text, unsafe_allow_html=True)
                    else:
                        result_text = f"{file_name}: No se encontraron resultados."
                elif model_choice == 'YoloV8 - ★★★★★ - 97.9%':
                    result_text = modelo_yolo(image)  # Asume que modelo_yolo devuelve un string con el resultado
                    predictions[file_name]=result_text
                elif model_choice == 'VGG-16 - ★★★☆☆ - 75%':
                    pass
                elif model_choice == 'RestNet - ★★★★☆ - 90%':
                    pass
            # Mostrar las predicciones debajo de cada imagen
    for i, prediction_text in enumerate(predictions):
        col = cols_in_row[i // cols_per_row][i % cols_per_row]
        with col:
            st.caption(prediction_text)



if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

st.write(f"dict predictions: {predictions}")
if uploaded_files:
    user_question = st.text_input("Escribe tu pregunta sobre la imagen para GPT-4:")
    if user_question and st.button('Enviar pregunta'):
        try:
            image_data_list = upload_multiple_files(uploaded_files)
            if 'messages' not in st.session_state:
                print(f"dict predictions: {predictions}")
                st.session_state['messages'] = analyze_image_with_gpt4(image_data_list, user_question, predictions)
            if user_question:
                with st.spinner("generando respuesta..."):
                    messages = st.session_state['messages']
                    messages = update_chat(messages, "user", user_question)
                    response = get_chatgpt_response(messages)
                    messages = update_chat(messages, "assistant", response)
                    st.session_state.past.append(user_question)
                    st.session_state.generated.append(response)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
            with st.expander("Show Messages"):
                st.write(messages)
        except Exception as e:
            st.error("Error al procesar la solicitud: " + str(e))
