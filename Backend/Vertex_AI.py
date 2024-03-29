from Backend.predict_api import predict_image_classification_sample

def predict_vertex(uploaded_file):
    # Guardar la imagen en un archivo temporal para Vertex AI
    with open("temp_image.jpg", "wb") as file:
        file.write(uploaded_file.getbuffer())

    # Llamar a la función de predicción de Vertex AI y capturar la respuesta
    prediction_response = predict_image_classification_sample(
    project="309967433708",
    endpoint_id="8459565498294599680",
    location="us-central1",
    filename="temp_image.jpg")
    return prediction_response
