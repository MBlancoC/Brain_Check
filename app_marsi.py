import streamlit as st
from PIL import Image
import openai
import base64
import io

st.title('ChatBot con GPT-4 Vision')

# Configuración de la clave API de OpenAI
# KEY Marsi

openai.api_key = "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"

# Subida de archivos y visualización de imagen
uploaded_file = st.file_uploader("Carga tu imagen aquí", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Convertir la imagen a base64 para la API de OpenAI
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Entrada de texto del usuario
    user_question = st.text_input("Introduce tu pregunta sobre la imagen:")

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
