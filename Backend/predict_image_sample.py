from predict_api import predict_image_classification_sample

# Define los parámetros
project_id="309967433708"
endpoint_id="8459565498294599680"
location="us-central1"
filename = "path_to_your_image_file.jpg"  # Asegúrate de proporcionar la ruta correcta

# Llamar a la función
predict_image_classification_sample(project_id, endpoint_id, location, filename)
