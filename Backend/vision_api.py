from openai import OpenAI
import Backend.parametros as p
from PIL import Image
import io
import base64

import openai
#from openai import OpenAI
from PIL import Image
import io
import base64

def analyze_image_with_gpt4(image_data_list, question, predictions):
    image_contents = [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"} for image_data in image_data_list]
    messages = [
        {
            "role": "system",
            "content": f"""
        #CONTEXTO#
        Asume un rol de radiólogo altamente capacitado y experimentado, especializado en el análisis de resonancias magnéticas cerebrales (MRI). Tu experiencia incluye la segmentación de estructuras cerebrales y la identificación de anomalías, particularmente tumores cerebrales.
        Tienes la capacidad de analizar escaneos de MRI, detectar la presencia de tumores y precisar su ubicación dentro del cerebro.
        Se te entregará un documento en formato diccionario con los resultados preliminares de las imágenes respecto de si tienen o no tumor. El diccionario posee el siguiente formato:
        <DICCIONARIO>: (Nombre: "57.jpg", Resultado: "SÍ tiene tumor")
        El diccionario es el siguiente:
        {predictions}
        #OBJETIVO#
        Utiliza tus conocimientos para asistir en discusiones educativas y orientadas a la únicamente a la investigación. Cuando se presenten una imagen o más imágenes de MRI cerebral,  analízalas para detectar cualquier tumor, especificando el tipo de tumor (si es posible) y su ubicación en el cerebro. Luego, participa en una discusión detallada e informativa con el usuario, quien es otro doctor especialista en radiología, respondiendo a sus preguntas basándote en el análisis.
        #ESTILO#
        Inicialmente responde entregando un diagnóstico completo de todas las imágenes,junto con el diccionario, enfocándote principalmente en aquellas que sí poseen tumores dado el listado otorgado.
        #TONO#
        Utiliza un tono académico
        #AUDIENCIA#
        Tu contenido será para investigadores, radiólogos, doctores y científicos expertos en la materia
        #RESPUESTA#
        Tu respuesta debe ser de MÁXIMO 150 palabras. No excedas ese límite en cada una de tus respuestas
        """
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}] + image_contents
        }
    ]


    #response = client.chat.completions.create(
       #model="gpt-4-vision-preview",
        #messages=messages,
       # max_tokens=500,
    #)
    #return response.choices[0].message.content
    return messages

def upload_multiple_files(uploaded_files):
    image_data_list = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_list.append(image_data)

    return image_data_list

    #if image_data_list and question:
        #return get_chatgpt_response(analyze_image_with_gpt4(image_data_list, question))
    #else:
        #return None

def get_chatgpt_response(messages):
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1000,
    )
    return response.choices[0].message.content

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages
