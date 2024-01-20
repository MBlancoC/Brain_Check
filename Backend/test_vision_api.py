# Suponiendo que este código se encuentra en test_vision_api.py dentro de la carpeta Backend

from vision_api import analyze_image_with_gpt4
import os
import base64

# Define la ruta a la carpeta de imágenes.
images_folder = 'Data/Images/p1'

# Lista para almacenar las rutas a las imágenes.
test_image_paths = []

# Lista para almacenar los datos de las imágenes en base64.
image_data_list = []

# Rellenamos la lista con todas las imágenes en la carpeta p1 y las codificamos en base64.
for filename in os.listdir(images_folder):
    if filename.lower().endswith((".jpg", ".jpeg")):  # Asegúrate de que la extensión coincida con tus archivos de imagen.
        image_path = os.path.join(images_folder, filename)
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            image_data_list.append(image_data)

# Formulamos una pregunta de prueba.
test_question = "Describe lo que veas en estas imágenes."

# Llamamos a la función analyze_image_with_gpt4 directamente.
response = analyze_image_with_gpt4(image_data_list, test_question)

# Imprimimos la respuesta.
print(response)
