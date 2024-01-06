import streamlit as st
from PIL import Image
from io import BytesIO
from Backend.predict_api import predict_image_classification_sample

st.title("Brain Check")

# Subida de archivo
uploaded_file = st.file_uploader("Carga tu imagen aquí", type=["jpg", "png"])
if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Guardar la imagen en un archivo temporal
    with open("temp_image.jpg", "wb") as file:
        file.write(uploaded_file.getbuffer())

    # Llamar a la función de predicción y capturar la respuesta
    prediction_response = predict_image_classification_sample(
        project="263184688391",
        endpoint_id="2305326238748639232",
        location="us-central1",
        filename="temp_image.jpg"
    )

    # Mostrar los resultados en la aplicación
    if prediction_response:
        for prediction in prediction_response:
            if 'displayNames' in prediction and 'confidences' in prediction and 'ids' in prediction:
                for displayName, id, confidence in zip(prediction['displayNames'], prediction['ids'], prediction['confidences']):
                    st.write(f"Resultado: {displayName}, Confianza: {confidence:.2f}")
